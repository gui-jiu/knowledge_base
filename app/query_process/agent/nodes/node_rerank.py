from typing import List, Dict, Any

from app.utils.reranker_http_utils import rerank_documents
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


# 动态 TopK 硬上限
RERANK_MAX_TOPK: int = 10
# 最小 TopK
RERANK_MIN_TOPK: int = 3
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
    """

    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取 query
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 2. 合并多源文档
        merged_multi_docs = self._merge_multi_source_docs(state)

        # 3. Rerank 精排
        reranked_docs = self._rerank_merged_docs(user_query, merged_multi_docs)

        # 4. 动态 TopK 截取
        cutoff_docs = self._cliff_cutoff(reranked_docs)

        state['reranked_docs'] = cutoff_docs
        return state

    def _merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果"""
        final_docs = []

        for rrf_doc in (state.get('rrf_chunks') or []):
            if not isinstance(rrf_doc, dict):
                continue
            final_docs.append({
                "content": rrf_doc.get('content'),
                "title": rrf_doc.get('title'),
                "chunk_id": rrf_doc.get('chunk_id'),
                "url": "",
                "source": "local",
            })

        for web_doc in (state.get('web_search_docs') or []):
            if not isinstance(web_doc, dict):
                continue
            final_docs.append({
                "content": web_doc.get('snippet'),
                "title": web_doc.get('title'),
                "chunk_id": None,
                "url": web_doc.get('url'),
                "source": "web",
            })

        logger.info(f"收集到准备进行 Rerank 精排的文档 {len(final_docs)}")
        return final_docs

    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""
        if not merged_multi_docs:
            return []

        try:
            contents = [doc.get("content") or "" for doc in merged_multi_docs]
            rerank_scores = rerank_documents(user_query, contents)

            scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]

            sorted_score_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
            return sorted_score_docs

        except Exception as e:
            logger.error(f"Rerank 重排序失败: {str(e)}")
            return [{**doc, "score": None} for doc in merged_multi_docs]

    def _cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断"""
        if not ranked_docs:
            return []

        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        cutoff_pos = upper_bound

        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            if current_score is None or next_score is None:
                continue

            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                logger.debug(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break

        return ranked_docs[:cutoff_pos]
