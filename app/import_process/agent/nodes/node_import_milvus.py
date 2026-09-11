import json
from typing import Dict, Any, List

from pymilvus import DataType

from app.utils.milvus_utils import get_milvus_client
from app.core.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.utils.milvus_utils import escape_milvus_string


class NodeImportMilvus(NodeBase):
    """
    节点: 导入向量库 (node_import_milvus)
    将处理好的向量数据写入 Milvus 数据库。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
            2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
            3. 幂等清理：删除同item_name旧数据，避免重复存储
            4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
            5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用

        必要参数：task_id、chunks
        更新参数：chunks字段回填chunk_id

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = self._step_1_check_input(state)

        # 步骤2：Milvus客户端连接+集合准备（自动建表）
        client = self._step_2_prepare_collection(vector_dimension)

        # 步骤3：幂等性处理 - 清理同item_name旧数据
        self._step_3_clean_old_data(client, chunks_json_data)

        # 步骤4：批量插入数据+主键chunk_id回填
        updated_chunks = self._step_4_insert_data(client, chunks_json_data)

        # 步骤5：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks

        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 切片包含sparse_vector核心字段
        :param state: 流程状态对象
        :return: (校验通过的切片列表, 稠密向量维度)
        """

        # 校验1：chunks非空
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("核心参数chunks为空或非列表类型")

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise ValueError("错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含sparse_vector字段
        if 'sparse_vector' not in first_chunk:
            raise ValueError("错误: 数据中缺失sparse_vector字段")

        # 提取向量维度和商品名称
        vector_dimension = len(first_chunk['dense_vector'])
        item_name = first_chunk.get('item_name', '未知商品名')
        logger.info(f"Milvus入库校验通过，待入库切片数：{len(chunks)} | 向量维度：{vector_dimension} | 商品名称：{item_name}")

        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        步骤2：Milvus客户端连接+集合准备
        :param vector_dimension: 稠密向量维度
        :return: MilvusClient 实例
        """

        # 1、从配置读取切片集合名称
        collection_name = milvus_config.chunks_collection

        # 2、配置缺失校验
        if not collection_name:
            logger.error("Milvus集合名称未配置：CHUNKS_COLLECTION为空")
            raise ValueError("未配置CHUNKS_COLLECTION集合名称")

        # 3、获取 Milvus 单例客户端
        client = get_milvus_client()
        if not client:
            logger.error("Milvus客户端获取失败：get_milvus_client()返回空")
            raise ValueError("Milvus 连接失败：get_milvus_client() 返回空")

        # 4、集合不存在则自动创建
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus集合{collection_name}不存在，开始自动创建Schema和索引")
            self._create_collection(client, collection_name, vector_dimension)
        else:
            logger.info(f"Milvus集合{collection_name}已存在，直接复用")

        return client

    def _create_collection(self, client, collection_name: str, vector_dimension: int):
        """
        辅助函数：Milvus集合+索引自动创建
        :param client: MilvusClient实例
        :param collection_name: 集合名称
        :param vector_dimension: 稠密向量维度
        """

        # 1. 创建Schema：自增主键+支持动态字段
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 2. 新增字段：业务字段+主键+双向量字段
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)

        # 3. 构建索引参数
        index_params = client.prepare_index_params()
        # 稠密向量索引：AUTOINDEX自动选最优索引类型+余弦相似度
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+内积
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 4. 创建集合
        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        logger.info(f"Milvus集合创建成功：{collection_name}，向量维度：{vector_dimension}")

    def _step_3_clean_old_data(self, client, chunks_json_data: List[Dict[str, Any]]):
        """
        步骤3：幂等性处理 - 基于item_name清理旧数据
        :param client: MilvusClient实例
        :param chunks_json_data: 待入库的切片列表
        """
        # 提取并去重item_name
        item_names = sorted({
            str(x.get("item_name", "")).strip()
            for x in chunks_json_data or []
            if str(x.get("item_name", "")).strip()
        })
        # 无有效item_name则跳过清理
        if not item_names:
            logger.warning("Milvus幂等性清理跳过：切片中无有效item_name")
            return
        # 多item_name提示日志
        if len(item_names) > 1:
            logger.warning(f"Milvus幂等性清理：本次检测到多个item_name，将逐个清理：{item_names}")

        # 遍历item_name，逐个清理旧数据
        for i_name in item_names:
            self._clear_chunks_by_item_name(client, i_name)

    def _clear_chunks_by_item_name(self, client, item_name: str):
        """
        内部核心函数：根据item_name删除Milvus中的旧切片数据
        :param client: MilvusClient实例
        :param item_name: 要清理的商品名称
        """
        try:
            # 1. 商品名称安全转义
            safe_item_name = escape_milvus_string(item_name)
            filter_expr = f'item_name == "{safe_item_name}"'

            # 2. 执行删除操作
            client.delete(collection_name=milvus_config.chunks_collection, filter=filter_expr)
            logger.info(f"Milvus幂等性清理完成：成功删除item_name={item_name}的旧数据")

        except Exception as e:
            logger.error(f"Milvus幂等性清理失败：item_name={item_name} | 错误：{str(e)}", exc_info=True)
            raise ValueError(f"幂等清理失败（item_name={item_name}）: {e}")

    def _step_4_insert_data(self, client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        步骤4：批量插入切片数据到Milvus+主键回填
        :param client: MilvusClient实例
        :param chunks_json_data: 待入库的切片列表
        :return: 回填了chunk_id的切片列表
        """
        # 1. 预处理数据：确保 part 字段存在，移除可能冲突的 chunk_id
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()
            # 移除手动 chunk_id，避免与自增主键冲突
            item_copy.pop("chunk_id", None)
            # 补充 part 字段
            if "part" not in item_copy:
                item_copy["part"] = 0
            # 确保 part 是 int（Milvus INT8）
            try:
                item_copy["part"] = int(item_copy["part"])
            except (ValueError, TypeError):
                item_copy["part"] = 0
            data_to_insert.append(item_copy)

        logger.info(f"Milvus数据插入：准备{len(data_to_insert)}条切片数据，开始批量插入")

        # 2. 执行批量插入
        insert_result = client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get('insert_count', 0)
        logger.info(f"Milvus数据插入完成：成功插入{insert_count}条数据")

        # 3. 主键回填
        inserted_ids = insert_result.get('ids', [])
        if inserted_ids:
            logger.info(f"Milvus主键回填：开始将{len(inserted_ids)}个自增chunk_id回填到切片")
            for idx, item in enumerate(chunks_json_data):
                item['chunk_id'] = str(inserted_ids[idx])
            logger.info("Milvus主键回填完成：所有切片已绑定chunk_id")

        return chunks_json_data


if __name__ == "__main__":
    import os
    from app.import_process.agent.state import create_default_state
    from app.core.paths import PROJECT_ROOT

    # 组装文件的绝对路径
    chunks_path = PROJECT_ROOT / "output/hak180产品安全手册/chunks_with_vector.json"
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
    node_import_milvus = NodeImportMilvus()
    final_state = node_import_milvus(init_state)
