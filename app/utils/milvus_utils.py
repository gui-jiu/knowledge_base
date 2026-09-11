from functools import lru_cache

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker

from app.core.logger import logger
from app.core.milvus_config import milvus_config


@lru_cache(maxsize=1)
def get_milvus_client():
    """
    获取 Milvus 客户端（单例，复用连接）
    使用 MilvusClient（高层 API），支持 has_collection/create_collection/insert/delete 等操作
    """
    try:
        client = MilvusClient(uri=milvus_config.milvus_url)
        # 验证连接：尝试列出集合
        client.list_collections()
        logger.info(f"Milvus 连接成功：{milvus_config.milvus_url}")
        return client
    except Exception as e:
        logger.error(f"Milvus 连接失败：{e}")
        return None


def escape_milvus_string(text: str) -> str:
    """
    转义 Milvus 过滤表达式中的特殊字符
    防止商品名称含引号/反斜杠等导致 filter 表达式解析失败
    """
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def create_hybrid_search_requests(dense_vector, sparse_vector, expr=None, limit: int = 10):
    """
    构造 Milvus 混合检索请求对象列表（稠密向量 + 稀疏向量）
    :param dense_vector: 稠密向量列表
    :param sparse_vector: 稀疏向量字典 {索引: 权重}
    :param expr: 过滤表达式（如 item_name in ["xxx"]）
    :param limit: 每路召回数量
    :return: [AnnSearchRequest(稠密), AnnSearchRequest(稀疏)]
    """
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        limit=limit,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit,
        expr=expr,
    )
    return [dense_req, sparse_req]


def hybrid_search(client, collection_name: str, reqs, ranker_weights=(0.8, 0.2),
                  norm_score: bool = True, limit: int = 5, output_fields=None):
    """
    执行 Milvus 混合检索（稠密 + 稀疏加权融合）
    :param client: MilvusClient 实例
    :param collection_name: 集合名
    :param reqs: create_hybrid_search_requests 返回的请求列表
    :param ranker_weights: 稠密/稀疏权重
    :param norm_score: 是否归一化评分
    :param limit: 最终返回数量
    :param output_fields: 返回的业务字段
    :return: 检索结果列表（list of list of dict）
    """
    if output_fields is None:
        output_fields = ["chunk_id", "content", "item_name"]

    ranker = WeightedRanker(ranker_weights[0], ranker_weights[1])

    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=limit,
        output_fields=output_fields,
    )

    # MilvusClient 返回 list[list[dict]]，每个 hit 形如 {"id":..., "distance":..., "entity":{输出字段}}
    normalized = []
    for hits in results:
        norm_hits = []
        for hit in hits:
            if isinstance(hit, dict):
                # 优先取嵌套的 entity 字段（MilvusClient 标准格式）
                entity = hit.get("entity")
                if not isinstance(entity, dict):
                    entity = {k: v for k, v in hit.items()
                              if k not in ("id", "distance", "score", "entity")}
                entity = dict(entity)
                entity.setdefault("chunk_id", hit.get("id"))
                norm_hits.append({
                    "id": hit.get("id"),
                    "distance": hit.get("distance", hit.get("score")),
                    "entity": entity,
                })
            else:
                norm_hits.append(hit)
        normalized.append(norm_hits)

    return normalized
