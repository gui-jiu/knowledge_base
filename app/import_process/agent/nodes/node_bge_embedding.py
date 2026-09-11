import json
import os
from pathlib import Path
from typing import List, Dict

from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.utils.embedding_utils import generate_embeddings


class NodeBgeEmbedding(NodeBase):
    """
    节点: 向量化 (node_bge_embedding)
    使用 BGE-M3 模型将文本切片转换为稠密 + 稀疏向量。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        主流程：
            1. 输入校验：验证chunks有效性
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态

        必要参数：task_id、chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据校验
        texts_to_embed = self._step_1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(texts_to_embed)

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(output_data)} 条文本切片 ---")

        return state

    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        向量化前置步骤1：输入数据有效性校验
        :param state: 流程全局状态对象
        :return: 校验通过的文本切片列表
        """

        # 1、从状态中提取切片数据
        texts_to_embed = state.get("chunks")

        # 2、校验：必须是非空列表
        if not isinstance(texts_to_embed, list) or not texts_to_embed:
            logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
            raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")

        logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")
        return texts_to_embed

    def _step_2_generate_embeddings(self, texts_to_embed: List[Dict]) -> List[Dict]:
        """
        向量化核心步骤2：批量生成稠密/稀疏双向量
        :param texts_to_embed: 文本切片列表
        :return: 带向量字段的文本切片列表
        """

        # 初始化结果列表，存储带向量的切片数据
        output_data = []
        # 批次大小配置：平衡内存占用和处理效率
        batch_size = 5

        # 按批次遍历，避免一次性处理过多数据导致内存溢出
        total = len(texts_to_embed)
        for i in range(0, total, batch_size):
            # 截取当前批次的切片，最后一批自动适配剩余数量
            batch_texts = texts_to_embed[i:i + batch_size]
            # 计算当前批次的起止索引，用于日志展示
            start_idx, end_idx = i + 1, min(i + len(batch_texts), total)

            try:
                # 构造模型输入文本：拼接商品名 + 切片内容，增强核心特征
                input_texts = []
                for doc in batch_texts:
                    item_name = doc.get("item_name", "")
                    content = doc.get("content", "")
                    # 有商品名则拼接（换行分隔），无则直接使用内容
                    text = f"{item_name}\n{content}" if item_name else content
                    input_texts.append(text)

                # 调用封装函数生成批量向量
                docs_embeddings = generate_embeddings(input_texts)
                if not docs_embeddings:
                    error_msg = f"第{start_idx}-{end_idx}条切片：BGE-M3模型返回空结果，无法生成向量"
                    logger.exception(error_msg)
                    raise RuntimeError(error_msg)

                # 为当前批次每个切片绑定对应向量
                for j, doc in enumerate(batch_texts):
                    item = doc.copy()
                    item["dense_vector"] = docs_embeddings["dense"][j]
                    item["sparse_vector"] = docs_embeddings["sparse"][j]
                    output_data.append(item)

                logger.info(f"第{start_idx}-{end_idx}条切片：双向量生成成功")

            except Exception as e:
                error_msg = f"第{start_idx}-{end_idx}条切片：向量生成失败。错误原因：{str(e)}"
                logger.exception(error_msg)
                raise RuntimeError(error_msg) from e

        return output_data


if __name__ == "__main__":
    # 获取项目所在路径
    from app.import_process.agent.state import create_default_state
    from app.core.paths import PROJECT_ROOT

    # 组装文件的绝对路径（使用 Path 对象，支持 read_text 方法）
    chunks_path = Path(PROJECT_ROOT) / "output/hak180产品安全手册/chunks_with_item_name.json"
    if not chunks_path.exists():
        chunks_path = Path(PROJECT_ROOT) / "output/hak180产品安全手册/chunks.json"
    # 读取切片
    chunks_json = chunks_path.read_text(encoding="utf-8")
    # 将json字符串chunks转成列表
    chunks = json.loads(chunks_json)
    # 当前节点图状态初始值
    init_state = create_default_state(
        task_id="task_001",
        chunks=chunks
    )
    # 执行节点的业务调用
    node_bge_embedding = NodeBgeEmbedding()
    final_state = node_bge_embedding(init_state)

    # 备份
    json_path = os.path.join(PROJECT_ROOT, "output", "hak180产品安全手册", "chunks_with_vector.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_state["chunks"], f, ensure_ascii=False, indent=2)
    logger.info(f"Chunk结果备份成功，备份文件路径：{json_path}")
