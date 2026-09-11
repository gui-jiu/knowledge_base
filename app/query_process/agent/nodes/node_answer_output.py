import re
from typing import List, Tuple, Dict

from app.utils.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.utils.llm_utils import get_llm_client
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import set_task_result

MAX_CONTEXT_CHARS = 12000


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案输出
    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """

    name: str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 阶段一：检查 answer 是否已存在
        answer_exists = self._step_1_check_answer(state)

        # 阶段二 & 三：构建 Prompt 并生成答案
        if not answer_exists:
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt
            self._step_3_generate_response(state, prompt)

        # 提取图片 URL
        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])

        # 阶段四：写入 MongoDB
        if state.get("answer"):
            logger.info("---写入MongoDB历史记录---")
            self._step_4_write_history(state, image_urls=image_urls)

        # 阶段五：发送 final 事件
        logger.info(f"---发送 final 事件---图片为：{image_urls}")
        if state.get("is_stream"):
            push_to_session(
                state['session_id'],
                SSEEvent.FINAL,
                {
                    "answer": state["answer"],
                    "status": "completed",
                    "image_urls": image_urls,
                },
            )

        logger.info("---node_answer_output 节点处理结束---")
        return state

    def _step_1_check_answer(self, state) -> bool:
        """检查 state 中是否已有 answer"""
        answer = state.get("answer")
        is_stream = state.get("is_stream")
        if answer:
            if is_stream:
                logger.info("---Step 1: 发现已有答案，执行流式推送---")
                push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
            else:
                set_task_result(state["session_id"], "answer", answer)
            return True
        return False

    def _step_2_construct_prompt(self, state: QueryGraphState) -> str:
        """构建 Prompt"""
        char_budget = MAX_CONTEXT_CHARS

        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = [str(n) for n in (state.get("item_names") or []) if n]

        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        item_names_str = ", ".join(item_names) if item_names else "无指定商品"

        prompt = load_prompt(
            "answer_out",
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        logger.info(f"组装后的提示词为：{prompt[:300]}...")
        return prompt

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content")
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field)).strip()
                if field_value and field_value != "None":
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + (content or "")

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _step_3_generate_response(self, state: QueryGraphState, prompt: str) -> QueryGraphState:
        """调用 LLM 生成答案，支持流式输出"""
        logger.info("---Step 3: 开始生成回答 (LLM Generation)---")
        llm = get_llm_client()

        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        if is_stream:
            logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
            final_text = ""
            try:
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        final_text += delta
                        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
                logger.info(f"流式输出完成，总长度: {len(final_text)}")
            except Exception as e:
                logger.error(f"流式生成出错: {e}", exc_info=True)
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
            state["answer"] = final_text
        else:
            logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = llm.invoke(prompt)
                content = response.content
                state["answer"] = content
                set_task_result(session_id, "answer", content)
                logger.info(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                logger.error(f"生成回答出错: {e}", exc_info=True)
                state["answer"] = "抱歉，生成回答时出现错误。"

        return state

    def _extract_images_from_docs(self, docs) -> List[str]:
        """从文档列表中提取图片 URL"""
        images = []
        seen = set()
        if not docs:
            return []

        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        logger.info(f"开始提取图片，待处理文档数: {len(docs)}")

        for i, doc in enumerate(docs):
            url = (doc.get("url") or "").strip()
            if url:
                if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                    if url not in seen:
                        seen.add(url)
                        images.append(url)

            text = (doc.get("content") or "").strip()
            if text:
                matches = md_img_pattern.findall(text)
                for img_url in matches:
                    img_url = img_url.strip()
                    if img_url and img_url not in seen:
                        seen.add(img_url)
                        images.append(img_url)

        logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
        return images

    def _step_4_write_history(self, state: QueryGraphState, image_urls=None) -> QueryGraphState:
        """把本轮答案写入 MongoDB history"""
        session_id = state.get("session_id", "default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []

        try:
            if answer:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                    image_urls=image_urls,
                )
        except Exception as e:
            logger.error(f"写入Mongo历史记录失败: {e}")

        return state
