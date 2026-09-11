import asyncio
import json
import threading

from app.core.bailian_mcp_config import mcp_config
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


def _run_async_safely(coro):
    """
    安全执行协程：
    - 若当前线程没有运行中的事件循环 → 直接 asyncio.run
    - 若有（如在 FastAPI async 端点内）→ 在独立线程中运行，避免嵌套事件循环报错
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container = {}

    def _worker():
        result_container["result"] = asyncio.run(coro)

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    return result_container.get("result")


class NodeWebSearchMcp(NodeBase):
    """
    节点功能：调用百炼 MCP 联网搜索服务，补充实时信息
    """

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query", "")
        docs = []

        # 未配置 MCP 地址则跳过（优雅降级）
        if not mcp_config.mcp_base_url:
            logger.warning("未配置 MCP_DASHSCOPE_BASE_URL，跳过联网搜索")
            return {"web_search_docs": []}

        if query:
            try:
                result = _run_async_safely(self._mcp_call(query))
                if result and getattr(result, "content", None):
                    pages = json.loads(result.content[0].text).get("pages") or []
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()
                        if not snippet:
                            continue
                        docs.append({"title": title, "url": url, "snippet": snippet})
                    logger.info(f"MCP 搜索结果: {docs}")
            except Exception as e:
                logger.error(f"MCP 联网搜索失败: {e}")

        if docs:
            return {"web_search_docs": docs}
        return {"web_search_docs": []}

    async def _mcp_call(self, query):
        from agents.mcp import MCPServerStreamableHttp

        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            await search_mcp.cleanup()
