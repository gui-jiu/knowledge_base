import json

import requests

from app.core.bailian_mcp_config import mcp_config
from app.query_process.agent.node_base import NodeBase
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState


class NodeWebSearchMcp(NodeBase):
    """
    节点功能：调用百炼 MCP 联网搜索服务，补充实时信息

    使用 MCP Streamable HTTP 协议（原生 JSON-RPC）实现：
      1. initialize（初始化握手）
      2. notifications/initialized（确认初始化）
      3. tools/call 调用 bailian_web_search 工具
    """

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query", "")

        # 未配置 MCP 地址则跳过（优雅降级）
        if not mcp_config.mcp_base_url:
            logger.warning("未配置 MCP_DASHSCOPE_BASE_URL，跳过联网搜索")
            return {"web_search_docs": []}

        if not query:
            return {"web_search_docs": []}

        docs = []
        try:
            pages = self._call_web_search(query)
            for item in pages:
                snippet = (item.get("snippet") or "").strip()
                url = (item.get("url") or "").strip()
                title = (item.get("title") or "").strip()
                if not snippet:
                    continue
                docs.append({"title": title, "url": url, "snippet": snippet})
            logger.info(f"MCP 联网搜索完成，返回 {len(docs)} 条结果")
        except Exception as e:
            logger.error(f"MCP 联网搜索失败: {e}")

        return {"web_search_docs": docs}

    def _call_web_search(self, query: str):
        """执行 MCP 协议调用，返回 pages 列表"""
        url = mcp_config.mcp_base_url
        headers = {
            "Authorization": f"Bearer {mcp_config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        session = requests.Session()

        # 1. initialize 握手
        session.post(url, headers=headers, json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "knowledge-base", "version": "1.0"},
            },
        }, timeout=15)

        # 2. 确认初始化
        session.post(url, headers=headers, json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }, timeout=15)

        # 3. 调用搜索工具
        resp = session.post(url, headers=headers, json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "bailian_web_search",
                "arguments": {"query": query, "count": 5},
            },
        }, timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(f"MCP 调用失败，状态码：{resp.status_code}")

        result = resp.json().get("result", {})
        contents = result.get("content") or []
        if not contents:
            return []

        # content[0].text 是 JSON 字符串，内含 pages 数组
        text = contents[0].get("text", "")
        data = json.loads(text)
        return data.get("pages") or []
