from typing import List, Dict, Any, Tuple

from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE）进行加权融合排序，提高相关性
    """

    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 各路搜索结果（网络搜索由 rerank 节点处理）
        vector_search_chunks = state.get('embedding_chunks') or []
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        # 2. 为不同路设置权重
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
        }

        # 3. 构建 rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4. 计算 RRF 得分
        rrf_merge_results = self._rrf_merge(rrf_inputs, k=60, max_results=10)

        # 5. 获取 rrf_chunks
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围
        scores = [s for _, s in rrf_merge_results]
        if scores:
            logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新 state
        state['rrf_chunks'] = rrf_chunks
        return state

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """统一处理各路检索结果，提取 entity"""
        diff_path_result = []
        if not rrf_input:
            return []

        for doc in rrf_input:
            if not isinstance(doc, dict):
                continue
            entity = doc.get('entity')
            if not entity:
                continue
            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """利用 RRF 公式计算每个文档的总得分并排序"""
        chunk_scores = {}
        chunk_data = {}

        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get('chunk_id')
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                chunk_data.setdefault(chunk_id, doc)

        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:max_results] if max_results else sorted_results
