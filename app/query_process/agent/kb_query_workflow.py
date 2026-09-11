from typing import Optional

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.core.logger import logger
from app.query_process.agent.nodes.node_answer_output import NodeAnswerOutput
from app.query_process.agent.nodes.node_item_name_confirm import NodeItemNameConfirm
from app.query_process.agent.nodes.node_rerank import NodeRerank
from app.query_process.agent.nodes.node_rrf import NodeRrf
from app.query_process.agent.nodes.node_search_embedding import NodeSearchEmbedding
from app.query_process.agent.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from app.query_process.agent.nodes.node_web_search_mcp import NodeWebSearchMcp
from app.query_process.agent.state import QueryGraphState, create_default_state

load_dotenv()


class KBQueryWorkflow:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑
    """

    def __init__(self):
        self.workflow = StateGraph(QueryGraphState)
        self._init_nodes()
        self._register_nodes()
        self._setup_routes()
        self._compiled_app: Optional[object] = None

    def _init_nodes(self):
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)
        self.workflow.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)
        self.workflow.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由：有 answer 直接输出，否则进入多路检索"""
        if state.get("answer"):
            return "node_answer_output"
        return "node_multi_search"

    def _setup_routes(self):
        self.workflow.set_entry_point("node_item_name_confirm")

        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_multi_search": "node_multi_search",
            },
        )

        # 并发执行多路搜索
        self.workflow.add_edge("node_multi_search", "node_search_embedding")
        self.workflow.add_edge("node_multi_search", "node_search_embedding_hyde")
        self.workflow.add_edge("node_multi_search", "node_web_search_mcp")

        # 多路搜索结果合并
        self.workflow.add_edge("node_search_embedding", "node_join")
        self.workflow.add_edge("node_search_embedding_hyde", "node_join")
        self.workflow.add_edge("node_web_search_mcp", "node_join")

        # 合并 -> 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("node_join", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def compile(self):
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False):
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        return self._compiled_app.invoke(initial_state)

    @classmethod
    def create_and_run(cls, initial_state: QueryGraphState, stream: bool = False):
        workflow = cls()
        return workflow.run(initial_state, stream)


if __name__ == "__main__":
    initial_state = create_default_state(
        task_id="task_001",
        session_id="query_20260331_001",
        original_query="HAK180烫金机怎么调整转印温度？",
    )
    final_state = KBQueryWorkflow.create_and_run(initial_state)
    logger.info(f"查询完成，答案：{final_state.get('answer')}")
