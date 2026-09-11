from typing import List

from app.core.logger import logger
from app.core.reranker_config import reranker_config


def rerank_documents(query: str, documents: List[str], top_n: int = None) -> List[float]:
    """
    调用百炼 qwen3-rerank 模型对文档进行相关性打分
    :param query: 查询问题
    :param documents: 待打分文档内容列表
    :param top_n: 返回前 N 个（None 表示全部）
    :return: 与 documents 顺序对应的相关性分数列表
    """
    if not documents:
        return []

    if not reranker_config.api_key:
        logger.warning("未配置 DASHSCOPE_API_KEY，Rerank 跳过，返回默认分数")
        return [0.0] * len(documents)

    try:
        import dashscope

        dashscope.api_key = reranker_config.api_key

        response = dashscope.TextReRank.call(
            model=reranker_config.model,
            query=query,
            documents=documents,
            top_n=top_n or len(documents),
            return_documents=False,
            instruct=reranker_config.instruct,
        )

        if response.status_code != 200:
            logger.error(f"Rerank 调用失败：{response.status_code} {response.message}")
            return [0.0] * len(documents)

        # 结果按 index 对应原始文档
        scores = [0.0] * len(documents)
        for item in response.output.results:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(documents):
                scores[idx] = float(item.get("relevance_score", 0.0))

        return scores

    except Exception as e:
        logger.error(f"Rerank 调用异常：{e}")
        return [0.0] * len(documents)
