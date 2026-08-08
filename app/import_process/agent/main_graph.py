from dotenv import load_dotenv
from langchain_core.callbacks.manager import handle_event
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.core.logger import logger
from app.import_process.agent.nodes.node_bge_embedding import NodeBgeEmbedding
from app.import_process.agent.nodes.node_document_split import NodeDocumentSplit
from app.import_process.agent.nodes.node_entry import NodeEntry
from app.import_process.agent.nodes.node_import_milvus import NodeImportMilvus
from app.import_process.agent.nodes.node_item_name_recognition import NodeItemNameRecognition
from app.import_process.agent.nodes.node_md_img import NodeMdImg
from app.import_process.agent.nodes.node_pdf_to_md import NodePdfToMd
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state

load_dotenv()

# 1、初始化状态图，指定整个状态图中的状态类型
workflow = StateGraph(ImportGraphState)

# 2、注册所有的业务节点
# 2.1 创建节点
node_entry = NodeEntry()
node_pdf_to_md = NodePdfToMd()
node_md_img = NodeMdImg()
node_document_split = NodeDocumentSplit()
node_item_name_recognition = NodeItemNameRecognition()
node_bge_embedding = NodeBgeEmbedding()
node_import_milvus = NodeImportMilvus()
# 2.2 注册节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 3、设置工作流的入口节点
workflow.set_entry_point("node_entry")
# 等同于
# workflow.add_edge(START, "node_entry")

# 4、定义条件边
# 4.1 定义条件路由
def route_after_entry(state: ImportGraphState) -> str:
    # PDF导入
    if state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    # MD导入
    elif state["is_md_read_enabled"]:
        return "node_md_img"
    # 流程终止
    else:
        return END
# 4.2 注册条件边
workflow.add_conditional_edges(
    "node_entry", # 入口节点
    route_after_entry, #条件路由

    # 如果不执行后面的 print_ascii()的话，这句话可以省略掉
    {
        "node_pdf_to_md":"node_pdf_to_md",
        "node_md_img":"node_md_img",
        END:END
    }
)

# 5、注册顺序边
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 6、编译工作流
kb_import_app = workflow.compile()
# 测试
if __name__ == "__main__":

    # 1 初始化状态信息
    init_state = create_default_state(
        task_id="task_001",
        local_file_path="d:/abc.md"
    )

    # 2（invoke） 运行工作流
    # 在图的外部，仅在整个图的执行过程都结束之后，我们才能够拿到最终的状态
    # final_state = kb_import_app.invoke(init_state)
    # print(format_state(final_state))

    # 2（stream）运行工作流
    # 在图的外部，每执行完一个节点，就可以输出当前的state
    # for chunk in kb_import_app.stream(init_state):
    #     # chunk：字典
    #     logger.info(chunk.keys())
    #     logger.info(chunk.items())

    logger.info("输出图结构:")
    # 以下代码需要 uv add grandalf
    kb_import_app.get_graph().print_ascii()