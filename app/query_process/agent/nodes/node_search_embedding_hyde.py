from app.utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.core.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.utils.embedding_utils import generate_embeddings
from app.utils.llm_utils import get_llm_client
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        try:
            # 生成假设性文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            # 用"重写问题 + 假设文档"检索切片
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
                top_k=5,
            )

            return {
                "hyde_embedding_chunks": res[0] if res else [],
                "hyde_doc": hyde_doc,
            }

        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {}

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """利用大模型根据用户查询生成假设性文档"""
        logger.info("步骤1: 开始生成假设性文档")
        try:
            llm = get_llm_client()
            hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
            response = llm.invoke(hyde_prompt)
            hyde_doc = response.content
            logger.info(f"步骤1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
            return hyde_doc
        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise e

    def _step_2_search_embedding_hyde(self, rewritten_query, hyde_doc, item_names=None,
                                      req_limit=10, top_k=5, ranker_weights=(0.8, 0.2),
                                      norm_score=True,
                                      output_fields=("chunk_id", "content", "item_name")):
        """用"重写问题 + 假设性文档"生成 embedding 并检索"""
        try:
            combined_text = rewritten_query + " " + hyde_doc
            logger.info(f"步骤2: 拼接 Query + HyDE Doc, 总长度: {len(combined_text)}")

            embeddings = generate_embeddings([combined_text])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            collection_name = milvus_config.chunks_collection
            logger.info(f"步骤2: 准备在集合 '{collection_name}' 中执行混合检索")

            expr = None
            if item_names:
                quoted = ", ".join(f'"{v}"' for v in item_names)
                expr = f"item_name in [{quoted}]"

            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=req_limit,
            )

            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=ranker_weights,
                norm_score=norm_score,
                limit=top_k,
                output_fields=list(output_fields),
            )

            return res

        except Exception as e:
            logger.error(f"步骤2: 检索过程发生异常: {e}")
            raise e
