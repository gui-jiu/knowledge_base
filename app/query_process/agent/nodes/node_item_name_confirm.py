import json
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from app.utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.utils.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)
from app.core.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.utils.embedding_utils import generate_embeddings
from app.utils.llm_utils import get_llm_client
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        必要参数：session_id、original_query
        更新参数：history、rewritten_query、item_names、answer
        """
        # 步骤1：校验参数
        session_id, original_query = self._step_1_validate_param(state)
        logger.info("步骤1：参数校验通过")

        # 步骤2：获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        state["history"] = history

        # 步骤3：用户初始消息保存
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names", [])
        rewritten_query = extract_res.get("rewritten_query", original_query)
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        # 步骤5 & 6：搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 步骤7：检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 步骤8：写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:
        session_id = (state.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = (state.get("original_query") or "").strip()
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    def _step_4_extract_info(self, query, history) -> Dict:
        """利用LLM从当前问题及历史会话中提取商品名 item_names 并改写问题"""
        try:
            logger.info("步骤4：正在初始化 LLM 客户端...")
            client = get_llm_client(json_mode=True)

            history_text = ""
            for msg in history:
                history_text += f"{msg['role']}: {msg['text']}\n"
            logger.info(f"步骤4：历史上下文准备完成 (长度: {len(history_text)})")

            prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)

            messages = [
                SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
                HumanMessage(content=prompt),
            ]

            logger.info("步骤4：正在调用 LLM...")
            response = client.invoke(messages)
            content = response.content

            # 清洗代码块包裹
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")

            result = json.loads(content)
            logger.info(f"步骤4：解析 LLM 结果: {result}")

            if "item_names" not in result:
                result["item_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            return result

        except Exception as e:
            logger.error(f"步骤4：LLM 提取失败: {e}")
            return {"item_names": [], "rewritten_query": query}

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """把商品名向量化，在 Milvus 商品名集合中混合检索"""
        logger.info(f"步骤5：开始向量化并查询条目: {item_names}")
        results = []

        client = get_milvus_client()
        if not client:
            logger.error("连接 Milvus 失败")
            return results

        collection_name = milvus_config.item_name_collection
        embeddings = generate_embeddings(item_names)
        logger.info(f"步骤5：已生成 {len(item_names)} 个商品名的向量。开始 Milvus 搜索...")

        for i in range(len(item_names)):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5,
                )

                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"],
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append({
                            "item_name": hit.get("entity", {}).get("item_name"),
                            "score": hit.get("distance"),
                        })

                results.append({
                    "extracted_name": item_names[i],
                    "matches": matches,
                })

            except Exception as e:
                logger.error(f"步骤5：查询商品名 '{item_names[i]}' 时出错: {e}")

        return results

    def _step_6_align_item_names(self, query_results) -> dict:
        """根据评分对齐商品名，生成确认列表和候选列表"""
        confirmed_item_names: List[str] = []
        options: List[str] = []

        logger.info(f"步骤6：获得待处理的数据源：{query_results}")

        for res in query_results:
            extracted_name = (res.get("extracted_name", "") or "").strip()
            matches = res.get("matches", []) or []
            if not matches:
                continue

            high = [m for m in matches if (m.get("score") or 0) > 0.85]
            mid = [m for m in matches if (m.get("score") or 0) >= 0.6]

            if len(high) == 1:
                confirmed_item_names.append(high[0].get("item_name"))
                continue

            if len(high) > 1:
                picked = None
                if extracted_name:
                    for m in high:
                        if m.get("item_name") == extracted_name:
                            picked = m
                            break
                if not picked:
                    picked = high[0]
                confirmed_item_names.append(picked.get("item_name"))
                continue

            if len(mid) > 0:
                for m in mid[:5]:
                    options.append(m.get("item_name"))

        return {
            "confirmed_item_names": [n for n in set(confirmed_item_names) if n],
            "options": [n for n in set(options) if n],
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        """检查对齐结果，分3种分支更新state"""
        confirmed = align_result.get("confirmed_item_names", [])
        options = align_result.get("options", [])

        # 分支A：有确认商品名
        if confirmed:
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    mid = msg.get("_id")
                    if mid:
                        ids_to_update.append(str(mid))

            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)

            state["item_names"] = confirmed
            if (state.get("answer", "") or "").strip():
                state["answer"] = ""
            return state

        # 分支B：有候选商品名
        if options:
            options_str = "、".join(options[:3])
            state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            state["item_names"] = []
            return state

        # 分支C：无匹配
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        state["item_names"] = []
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """把本次处理的核心信息写入 MongoDB"""
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state["answer"],
                rewritten_query="",
                item_names=state.get("item_names", []),
            )

        save_chat_message(
            session_id=session_id,
            role="user",
            text=state["original_query"],
            rewritten_query=rewritten_query,
            item_names=state.get("item_names", []),
            message_id=message_id,
        )

        return state
