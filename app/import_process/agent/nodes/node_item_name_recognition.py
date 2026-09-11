import json
import os
from typing import List, Tuple, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from app.utils.milvus_utils import get_milvus_client, escape_milvus_string
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.utils.embedding_utils import generate_embeddings
from app.utils.llm_utils import get_llm_client
from app.core.milvus_config import milvus_config

# --- 配置参数 (Configuration) # --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

class NodeItemNameRecognition(NodeBase):
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph 核心节点：商品主体名称识别
        流程总览：
            1. 提取输入
            2. 构建大模型上下文
            3. 调用大模型识别商品名称
            4. 回填商品名称到状态和切片
            5. 生成商品名称的稠密/稀疏向量
            6. 将数据存入Milvus向量数据库

        必要参数：task_id、file_title, chunks
        更新参数：item_name

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)

        # 步骤4：回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)

        # 步骤5：为商品名称生成稠密/稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)

        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)


        # 打印识别结果
        logger.info(f"--- 识别完成: {item_name} ---")

        return state



    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        """
        步骤 1: 接收并校验流程输入
        核心作用：
            1. 从流程状态中提取文件标题、文本切片核心数据
            2. 基础数据类型校验，保证下游流程输入有效性
        依赖的状态数据（上游节点产出）：
            - state["file_title"]: 上游提取的文件标题
            - state["chunks"]: 文本切片列表
        返回值：
            Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
        """

        # 1、参数校验
        file_title = state.get("file_title", "")
        if not file_title:
            raise ValueError("核心参数file_title缺失")

        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("核心参数chunks为空或非列表类型")

        logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")
        return file_title, chunks



    def _step_2_build_context(self, chunks: List[Dict]) -> str:
        """
        步骤 2: 构造大模型商品名称识别的标准化上下文
        核心作用：
            1. 限制切片数量：仅取前k个切片，避免上下文过长
            2. 限制字符长度：总上下文字符限制，适配大模型输入上限
            3. 格式化内容：带序号的结构化格式，提升大模型识别精度
        参数说明：
            chunks: 文本切片列表
        返回值：
            str: 格式化后的上下文字符串
        """

        # 1、初始化变量
        # 存储格式化后的切片片段，保证上下文结构化
        parts: List[str] = []
        # 统计已拼接字符数，用于控制总长度不超限
        total_chars = 0

        # 2、遍历前k个切片，避免上下文过长
        for idx, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):

            # 2.1、提取切片标题和内容，去首尾空格，过滤无效字符
            chunk_title = chunk.get("title", "").strip()
            chunk_content = chunk.get("content", "").strip()

            # 2.2、结构化格式化切片：带序号+标题+内容，提升大模型识别效率
            piece = f"【切片{idx}】\n标题：{chunk_title} \n内容：{chunk_content}"
            parts.append(piece)

            # 2.3、累计字符数，包含分隔符
            total_chars += len(piece)

            # 2.3、总字符数超限时立即停止拼接，避免大模型输入超限
            if total_chars > CONTEXT_TOTAL_MAX_CHARS:
                logger.info(f"上下文总字数：{total_chars}，已超限（{CONTEXT_TOTAL_MAX_CHARS}），已停止拼接后续切片")
                break

        # 3、用空行分隔切片片段，拼接为最终上下文，去重空格
        context = "\n\n".join(parts).strip()

        # 4、二次截断，确保绝对不超限
        final_context = context[:CONTEXT_TOTAL_MAX_CHARS]
        logger.info(f"步骤2：上下文构建完成，最终长度{len(final_context)}字符")
        return final_context




    def _step_3_call_llm(self, file_title: str, context: str) -> str:
        """
        步骤 3: 调用大模型实现商品名称/型号精准识别
        核心逻辑：
            1. 上下文为空 → 直接返回file_title（兜底，无需调用大模型）
            2. 上下文非空 → 加载标准化prompt模板，构建大模型对话消息
            3. 调用大模型后对返回结果做清洗，过滤无效字符
            4. 大模型返回空/调用异常 → 均返回file_title兜底，保证流程不中断
        核心特性：
            - 提示词解耦：通过load_prompt加载本地模板，无需硬编码
            - 格式兼容：兼容不同LLM客户端返回格式，防止属性报错
            - 异常兜底：全异常捕获，大模型服务不可用时不影响主流程
        参数：
            file_title: 处理后的文件标题（异常/空值时的兜底值）
            context: 步骤2构建的结构化切片上下文（大模型识别的核心依据）
        返回值：
            str: 清洗后的商品名称（异常/空值时返回原始file_title）
        """

        logger.info("开始执行步骤3：调用大模型识别商品名称")

        # 1、上下文为空时，直接返回文件标题，跳过大模型调用
        if not context:
            logger.warning("上下文为空，跳过大模型调用，直接使用文件标题作为商品名称")
            return file_title

        try:
            # 2、加载商品名称识别prompt模板，动态传入文件标题和上下文
            human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)

            # 3、加载系统提示词，定义大模型角色（商品识别专家，仅返回纯结果）
            system_prompt = load_prompt("product_recognition_system")
            logger.debug(
                f"大模型调用提示词构建完成，系统提示词长度{len(system_prompt)}，人类提示词长度{len(human_prompt)}")

            # 4、获取大模型客户端（默认模型，返回纯文本）
            llm = get_llm_client()
            if not llm:
                logger.error("大模型客户端获取失败，使用文件标题兜底")
                return file_title

            # 5、标准化构建大模型对话消息：SystemMessage定义角色 + HumanMessage传递业务请求
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            # 6、调用大模型并获取返回结果
            resp = llm.invoke(messages)

            # 7、获取返回值：兼容不同 LLM 客户端返回格式：可配置的属性名列表
            content_attributes = ["content", "text", "message", "response"]
            for attr in content_attributes:
                item_name = getattr(resp, attr, None)
                if item_name:
                    item_name = item_name.strip()
                    break
                else:
                    item_name = ""

            # 8、清洗返回结果：过滤空格、换行、回车、制表符等无效字符
            item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

            # 9、清洗后结果为空，使用文件标题兜底
            if not item_name:
                logger.warning("大模型返回空内容，使用文件标题作为商品名称兜底")
                return file_title

            logger.info(f"步骤3：大模型识别商品名称成功，结果为：{item_name}")
            return item_name

        # 捕获所有异常：大模型调用超时、网络错误、格式错误等，均不中断主流程
        except Exception as e:
            logger.exception(f"步骤3：大模型调用失败，原因：{str(e)}")
            # 异常时返回文件标题兜底，保证流程继续执行
            return file_title




    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):
        """
        步骤 4: 回填商品名称到流程状态和所有文本切片
        核心作用：
            1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
            2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
            3. 状态同步：更新state中的chunks，确保切片修改全局生效
        设计思路：
            所有切片关联同一商品名称，保证后续向量入库、检索时的维度一致性
        参数：
            state: 流程状态对象（ImportGraphState），全局数据载体
            chunks: 校验后的文本切片列表
            item_name: 步骤3识别并清洗后的商品名称
        """
        # 1、将商品名称存入全局状态，供下游节点调用
        state["item_name"] = item_name

        # 2、遍历所有切片，为每个切片添加商品名字段，保证数据全链路一致
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 3、同步更新state中的切片列表，确保修改全局生效
        state["chunks"] = chunks

        logger.info(f"步骤4：商品名称回填完成，共为{len(chunks)}个切片添加item_name字段，值为：{item_name}")


    def _step_5_generate_vectors(self, item_name: str) -> Tuple[List, Dict]:
        """
        步骤 5: 为商品名称生成 BGE-M3 稠密 + 稀疏双向量（Milvus 向量检索核心）
        核心说明：
            - 稠密向量（dense_vector）：BGE-M3 固定 1024 维，记录文本深层语义信息
            - 稀疏向量（sparse_vector）：变长键值对，记录文本关键词/特征位置信息
        依赖工具：
            generate_embeddings：封装 BGE-M3 模型，批量生成双向量，兼容单条/批量输入
        参数：
            item_name: 步骤 3 识别的商品名称（非空，空值时直接返回空向量）
        返回值：
            Tuple[List, Dict]: (稠密向量列表，稀疏向量字典)，异常时抛出异常
        """
        logger.info(f"步骤 5：为商品名称 [{item_name}] 生成 BGE-M3 双向量")

        # 1、商品名称为空，直接抛出异常，因为后续向量检索必须依赖这两个向量
        if not item_name:
            error_msg = "商品名称为空，无法生成向量"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # 2、调用向量生成工具：传入列表支持批量生成，单条数据仍用列表保证格式统一
            vector_result = generate_embeddings([item_name])

            # 3、向量生成结果非空，才进行后续解析
            if vector_result and "dense" in vector_result and "sparse" in vector_result:
                # 稠密向量解析：取批量结果第一个，为 Python 列表（Milvus 存储要求）
                dense_vector = vector_result["dense"][0]
                # 稀疏向量解析：取批量结果第一个，CSR 矩阵解析为字典格式
                sparse_vector = vector_result["sparse"][0]
                logger.info("步骤 5：BGE-M3 稠密 + 稀疏向量生成成功")
            else:
                # 向量生成失败，抛出异常，阻止无向量数据入库
                error_msg = f"向量生成工具返回空结果，无法为 [{item_name}] 生成双向量"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        # 4、捕获所有异常并重新抛出，让上层知道向量生成失败
        except Exception as e:
            logger.error(f"步骤 5：向量生成失败，原因：{str(e)}", exc_info=True)
            raise  # 重新抛出异常，中断流程

        return dense_vector, sparse_vector

    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str, dense_vector,
                               sparse_vector):
        """
        步骤 6: 将商品名称、文件标题、双向量持久化到 Milvus 向量数据库
        核心逻辑：
            1. 配置校验：检查 Milvus 连接地址和集合名配置，缺失则跳过
            2. 客户端获取：获取单例 Milvus 客户端，连接失败则跳过
            3. 集合初始化：无集合则创建（定义 Schema+索引），有集合则直接使用
            4. 幂等性处理：删除同名商品数据，避免重复存储
            5. 数据插入：构造符合 Schema 的数据，非空向量才添加
            6. 集合加载：插入后强制加载集合，确保数据立即可查/Attu 可见
        索引设计：
            - 稠密向量：IVF_FLAT 索引 + 余弦相似度（COSINE），兼容性好，适合小数据量
            - 稀疏向量：SPARSE_INVERTED_INDEX 索引 + 内积（IP），稀疏向量专用，检索效率高
        参数：
            state: 流程状态对象，用于最终状态同步
            file_title: 处理后的文件标题
            item_name: 识别后的商品名称（主键去重依据）
            dense_vector: 步骤 5 生成的稠密向量（1024 维列表）
            sparse_vector: 步骤 5 生成的稀疏向量（字典格式）
        """

        # 1、从环境变量读取 Milvus 核心配置，与 MilvusConfig 配置类保持一致
        collection_name = milvus_config.item_name_collection

        # 2、配置缺失校验：任一配置为空则跳过 Milvus 存储，记录警告
        if not collection_name:
            logger.warning("Milvus 配置缺失 ITEM_NAME_COLLECTION，跳过数据保存")
            return

        logger.info(f"开始执行步骤 6：将商品名称 [{item_name}] 保存到 Milvus 集合 [{collection_name}]")

        try:
            # 3、获取 Milvus 单例客户端，连接失败则直接返回
            client = get_milvus_client()
            if not client:
                logger.error("无法获取 Milvus 客户端（连接失败），跳过数据保存")
                return

            # 4、集合初始化：不存在则创建（定义Schema+索引），存在则直接使用
            if not client.has_collection(collection_name=collection_name):
                logger.info(f"Milvus 集合 [{collection_name}] 不存在，开始创建 Schema 和索引")
                # 创建集合 Schema：自增主键 + 动态字段，适配灵活的数据存储
                schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
                # 添加自增主键字段：INT64 类型，唯一标识每条数据
                schema.add_field(
                    field_name="pk",
                    datatype=DataType.INT64,
                    is_primary=True,
                    auto_id=True
                )

                # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
                schema.add_field(
                    field_name="file_title",
                    datatype=DataType.VARCHAR,
                    max_length=65535
                )

                # 添加商品名字段：VARCHAR类型，最大长度65535，去重依据
                schema.add_field(
                    field_name="item_name",
                    datatype=DataType.VARCHAR,
                    max_length=65535
                )

                # 添加稠密向量字段：FLOAT_VECTOR，1024维（BGE-M3固定维度）
                schema.add_field(
                    field_name="dense_vector",
                    datatype=DataType.FLOAT_VECTOR,
                    dim=1024
                )
                # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR，变长
                schema.add_field(
                    field_name="sparse_vector",
                    datatype=DataType.SPARSE_FLOAT_VECTOR
                )

                # 构建索引参数：为向量字段创建索引，提升检索性能
                index_params = client.prepare_index_params()

                # 稠密向量索引：IVF_FLAT+COSINE，nlist=128（聚类数，平衡精度和速度）
                # 与其他索引对比（速览）
                """
                  索引	        精度          速度	        适用规模
                  FLAT	        100%          最慢	        ≤10 万
                  IVF_FLAT      高（≈98%）     中            10 万–1000 万
                  IVF_PQ	    中（≈90%）     快 	        千万–亿级
                  HNSW	        高	          最快	        全规模
                """
                index_params.add_index(
                    field_name="dense_vector",
                    index_name="dense_vector_index",
                    # IVF_FLAT（Inverted File with Flat） 是向量数据库中最常用的高精度、中等速度的近似最近邻（ANN）索引算法，核心是 “先聚类分桶、再桶内暴力精确检索”。
                    index_type="IVF_FLAT",
                    #  COSINE（余弦相似度） 文本语义检索中，不同长度的句子（如 “苹果手机” 和 “我想买苹果手机”）的向量长度不同，但方向一致，用余弦能精准匹配语义，忽略长度差异。
                    metric_type="COSINE",
                    # nlist：聚类数（桶数），通常设为 4×√N（N 为向量总数）
                    params={"nlist": 128}
                )

                # 稀疏向量索引
                index_params.add_index(
                    field_name="sparse_vector",
                    index_name="sparse_vector_index",
                    # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，
                    # 特点是大部分元素为 0，只有少数维度有值，是稀疏向量检索的标配索引类型。
                    index_type="SPARSE_INVERTED_INDEX",
                    # IP（内积，Inner Product）如果向量是 “文本语义向量 + 关键词权重”，长度代表文本与主题的关联强度，
                    # 此时用 IP 能同时体现 “语义匹配度” 和 “关联强度”。
                    metric_type="IP",
                    params={
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        # ↑ 使用 DAAT_MAXSCORE 算法（Dynamic And-And Threshold）
                        #   高效的稀疏检索算法，类似搜索引擎的倒排索引

                        "normalize": True,
                        # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度
                        #   结果范围在 0-1 之间，便于理解

                        "quantization": "none"
                        # ↑ 关闭量化，保持原始精度：模型生成的向量已经压缩的一半的精度了（BGE_FP16=1），这里就不再压缩了

                        # "quantization": "none" → 存储原始向量，不压缩
                        # "quantization": "sq8" → 存储压缩后的向量（8-bit 量化
                    }
                )

                # 创建集合：Schema + 索引参数
                client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
                logger.info(f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引")

            # 5、幂等性处理：删除同名商品数据，避免重复存储
            # 商品名称转义，防止特殊字符导致过滤表达式解析失败
            safe_item_name = escape_milvus_string(item_name)
            filter_expr = f'item_name=="{safe_item_name}"'

            # 6、执行删除操作
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(f"Milvus幂等性处理完成，已删除集合中[{item_name}]的历史数据")

            # 7、构造插入 Milvus 的数据：必须包含两个向量，否则无法检索
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            # 8、插入数据：列表格式支持批量插入，单条数据保持格式统一
            client.insert(collection_name=collection_name, data=[data])

            # 最终同步商品名称到全局状态
            state["item_name"] = item_name
            logger.info(f"步骤6：商品名称[{item_name}]成功存入Milvus集合[{collection_name}]")

        # 捕获所有Milvus操作异常：连接中断、入库失败、索引错误等，不中断主流程
        except Exception as e:
            logger.warning(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)