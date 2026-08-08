import json
from typing import Any


def format_state(state: dict) -> str:
    """
    将图状态格式化为适合日志输出的字符串。
    - 使用 json.dumps 转成格式化 JSON
    - ensure_ascii=False 让中文正常显示
    - 长列表（chunks/embeddings_content）只打印数量和前几条，避免日志爆炸
    """
    display = {}

    for key, value in state.items():
        if isinstance(value, list) and len(value) > 5:
            display[key] = f"[list] 共{len(value)}条，前3条: {value[:3]}"
        else:
            display[key] = value

    return json.dumps(display, ensure_ascii=False, indent=2)
