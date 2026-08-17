from langchain_openai import ChatOpenAI

from app.core.lm_config import lm_config


def get_llm_client(model: str = None, temperature: float = None):
    """
    获取 LLM / VL 模型客户端
    - model 为空 → 用默认 LLM（DeepSeek）
    - model 传入 VL 模型名（如 qwen-vl-plus）→ 自动切到百炼 base_url 和 key
    """
    if model and model == lm_config.vl_model:
        return ChatOpenAI(
            model=model,
            api_key=lm_config.vl_api_key,
            base_url=lm_config.vl_api_base,
            temperature=temperature if temperature is not None else lm_config.temperature,
        )

    return ChatOpenAI(
        model=model if model else lm_config.default_model,
        api_key=lm_config.api_key,
        base_url=lm_config.api_base,
        temperature=temperature if temperature is not None else lm_config.temperature,
    )
