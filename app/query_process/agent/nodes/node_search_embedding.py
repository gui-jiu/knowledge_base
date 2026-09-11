from app.utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.core.milvus_config import milvus_config
from app.utils.embedding_utils import generate_embeddings
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        try:
            # 1、用户问题和已确认商品名
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2、生成向量 (Dense + Sparse)
            logger.info("正在生成混合向量 (Embedding)...")
            embeddings = generate_embeddings([query])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            # 3、获取 Milvus 集合
            collection_name = milvus_config.chunks_collection
            logger.info(f"准备在集合 '{collection_name}' 中执行混合检索")

            # 4、构造过滤条件
            expr = None
            if item_names:
                quoted = ", ".join(f'"{v}"' for v in item_names)
                expr = f"item_name in [{quoted}]"
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            # 5、构造混合搜索请求
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=10,
            )

            # 6、执行混合检索
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                limit=5,
                output_fields=["chunk_id", "content", "item_name"],
            )

            logger.info(f"节点search_embedding处理成功 :{res}")
            return {"embedding_chunks": res[0] if res else []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}
