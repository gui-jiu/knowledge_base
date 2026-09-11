import os

from dotenv import load_dotenv

load_dotenv()


class MCPConfig:
    """百炼 MCP 联网搜索配置"""

    def __init__(self):
        # 复用百炼 DashScope 的 Key
        self.api_key: str = os.getenv("DASHSCOPE_API_KEY", os.getenv("VL_API_KEY", ""))
        self.mcp_base_url: str = os.getenv("MCP_DASHSCOPE_BASE_URL", "")


mcp_config = MCPConfig()
