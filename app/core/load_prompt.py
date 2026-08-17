from pathlib import Path

from app.core.paths import PROJECT_ROOT

_PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    """
    加载并格式化提示词文件
    :param name: 提示词文件名（不带 .prompt 后缀）
    :param kwargs: 占位符变量（如 root_folder、image_content）
    :return: 渲染后的提示词字符串
    """
    prompt_path = _PROMPTS_DIR / f"{name}.prompt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    return template.format(**kwargs)
