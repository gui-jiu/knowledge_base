import os
import base64
import re
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_core.messages import HumanMessage
from minio import Minio
from minio.deleteobjects import DeleteObject

from app.utils.minio_utils import get_minio_client
from app.core.lm_config import lm_config
from app.core.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.utils.llm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit


class NodeMdImg(NodeBase):
    """
    节点: 图片处理 (node_md_img)
    处理 Markdown 中的图片资源 (Image)。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_md_img"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        MD文件图片处理核心节点
        核心流程：
        1. 获取MD内容、文件路径、图片文件夹路径
        2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
        3. 调用多模态大模型为图片生成内容摘要
        4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
        5. 备份原MD文件，保存处理后的新MD文件并更新状态

        必要参数：task_id、md_path、md_content
        更新参数：md_path、md_content

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：初始化数据，获取MD核心信息
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        # 无图片文件夹，直接跳过图片处理逻辑
        if not images_dir.exists():
            logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")
            return state

        # 步骤2：扫描并筛选MD中引用的图片
        target_images = self._step_2_scan_images(md_content, images_dir)
        if not target_images:
            logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")
            return state

        # 步骤3：调用多模态大模型生成图片摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

        # 步骤4：上传图片至MinIO，替换MD图片路径并填充摘要
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        # 步骤5：备份并保存新MD文件
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        # 步骤6：更新state状态值
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        从全局状态中提取并初始化MD处理所需核心数据
        :param state: 流程全局状态对象
        :return: 三元组(MD文件内容, MD文件路径, 图片文件夹路径)
        :raise FileNotFoundError: 当状态中无有效MD文件路径时抛出
        """

        # 1、非空校验
        md_path = state.get("md_path", "").strip()
        if not md_path:
            raise ValueError("核心参数md_path缺失")

        # 2、路径转换
        md_path_obj = Path(md_path)

        # 3、检查PDF文件的有效性
        if not md_path_obj.exists():
            raise ValueError(f"MD文件不存在，绝对路径: {md_path_obj.absolute()}")

        # 4、优先使用状态中已存在的MD内容，无则从文件读取
        if not state["md_content"]:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
            logger.info(f"从文件读取MD内容完成，文件大小：{len(md_content)} 字符")
            state["md_content"] = md_content
        else:
            md_content = state["md_content"]
            logger.info(f"从全局状态获取MD内容完成，内容大小：{len(md_content)} 字符")

        # 5、组装图片文件夹路径：图片文件夹固定为MD文件同级的images目录
        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
        :param md_content: MD文件完整内容
        :param images_dir: 图片文件夹路径对象
        :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
        """

        # MinIO支持的图片格式集合（小写后缀，统一匹配标准）
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        target_images = []

        # 1、遍历图片文件夹
        for image_file in os.listdir(images_dir):

            # 1.1、过滤无效后缀
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in image_extensions:
                logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue

            # 1.2、组装图片完整路径并转成字符串
            img_path = str(images_dir / image_file)

            # 1.3、查找图片在MD中的引用上下文
            context = self._find_image_in_md(md_content, image_file)

            # 过滤MD中未引用的图片
            if not context:
                logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
                continue

            # 1.4、组装待处理图片元数据，取第一个匹配的图片上下文
            target_images.append((image_file, img_path, context))
            logger.info(f"图片加入待处理列表：{image_file}")

        logger.info(f"图片扫描完成，共筛选出待处理图片：{len(target_images)} 张")
        return target_images

    def _find_image_in_md(self, md_content: str, image_file: str, context_len: int = 100) -> Tuple[str, str]:
        """
        查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
        :param md_content: MD文件完整内容
        :param image_file: 图片文件名（含后缀）
        :param context_len: 上下文截取长度，默认前后各100字符
        :return: 每个图片的(上文, 下文)元组，无匹配则返回None
        """

        # 1、定义正则表达式
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

        # 2、找到1个匹配项即返回
        match = pattern.search(md_content)
        if not match:
            return None  # 没有找到

        # 3、截取匹配位置的上文和下文（防止索引越界）
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        # 打印图片上下文，便于调试
        logger.debug(f"图片[{image_file}]匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片[{image_file}]匹配到引用，下文：{post_text.strip()}")

        # 4、返回上下文元组
        return pre_text, post_text

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
        """
        步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """
        summaries = {}

        # 1、外部初始化双端队列，用于API速率限制，跨循环复用
        request_deque = deque()

        # 2、循环处理图片
        for img_file, image_path, context in target_images:

            # 2.1、速率限制
            apply_api_rate_limit(request_deque, max_requests=10, window_seconds=30)

            # 2.2、调用大模型生成图片摘要
            logger.info(f"开始生成图片摘要：{image_path}")
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)

        logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")
        return summaries

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        """
        调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
        生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
        :param image_path: 图片本地完整路径
        :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
        :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
        :return: 图片内容摘要（异常时返回默认值"图片描述"）
        """

        # 1、加载并渲染提示词（传入所有占位符对应的变量）
        prompt_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=image_content
        )

        # 2、将图片编码为Base64，适配多模态大模型输入要求
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        # 3. 构造LangChain标准多模态HumanMessage（兼容千问/OpenAI等视觉模型）
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    # 多模态核心：Base64编码图片数据
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        # 4. 获取LLM客户端
        lvm_client = get_llm_client(model=lm_config.vl_model)

        # 5. 调用大模型
        response = lvm_client.invoke(messages)

        # 6. 解析响应（LangChain统一返回content字段）
        summary = response.content.strip().replace("\n", "")
        logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")

        return summary

    def _step_4_upload_and_replace(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]],
                                   summaries: Dict[str, str], md_content: str) -> str:
        """
        步骤4：清理MinIO旧目录 → 批量上传新图片 → 合并摘要和URL → 替换MD内容并存为新文档
        :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param md_content: 原始MD文件内容
        :return: 图片引用替换后的新MD内容
        """

        # 获取MinIO客户端
        minio_client = get_minio_client()

        # 获取MinIO上传目录
        minio_img_dir = minio_config.minio_img_dir
        # 转换成MinIO上传目录：上传目录 + 文档主名（去除空格，避免路径问题）
        upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

        # 步骤1：清理该文档对应的MinIO旧目录
        self._clean_minio_directory(minio_client, upload_dir)

        # 步骤2：批量上传图片至MinIO，获取URL映射
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)

        # 步骤3：合并图片摘要和URL，过滤上传失败的图片
        image_info = self._merge_summary_and_url(summaries, urls)

        # 步骤4：替换MD内容中的本地图片引用为MinIO远程引用
        md_content = self._process_md_file(md_content, image_info)

        return md_content

    def _clean_minio_directory(self, minio_client: Minio, prefix: str) -> None:
        """
        幂等性清理MinIO指定目录下的所有旧文件，防止垃圾文件堆积
        幂等性：多次调用结果一致，无文件时不报错
        :param minio_client: 初始化完成的MinIO客户端对象
        :param prefix: MinIO目录前缀（要清理的目录路径）
        """
        try:
            # 列出指定前缀下的所有对象（递归遍历子目录）
            # 注意：prefix 前面不能有 /, 否咋无法找到待删除的文件
            objects_to_delete = minio_client.list_objects(
                bucket_name=minio_config.bucket_name,
                prefix=prefix,
                recursive=True
            )

            # 构造删除对象列表（列表推导式）
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                logger.info(f"开始清理MinIO旧文件，待删除文件数：{len(delete_list)}，目录：{prefix}")
                # 批量删除对象
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
                # 遍历删除错误信息，记录异常
                for error in errors:
                    logger.error(f"MinIO文件删除失败：{error}")

                logger.info("MinIO旧文件清理完成")
            else:
                logger.info(f"MinIO目录无旧文件，无需清理：{prefix}")
        except Exception as e:
            logger.error(f"MinIO目录清理失败：{prefix}，错误信息：{str(e)}")

    def _upload_images_batch(self, minio_client: Minio, upload_dir: str,
                             target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        :param minio_client: 初始化完成的MinIO客户端对象
        :param upload_dir: MinIO上传根目录
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
        """
        urls = {}
        # 元组解包
        for img_file, img_path, _ in target_images:
            # 构造MinIO对象名称
            object_name = f"{upload_dir}/{img_file}"
            logger.info(f"构造MinIO对象名称完成：{object_name}")
            # 上传单张图片并获取URL（海象运算符：表达式内赋值+判断）
            if img_url := self._upload_to_minio(minio_client, img_path, object_name):
                urls[img_file] = img_url

        logger.info(f"图片批量上传完成，成功上传{len(urls)}/{len(target_images)}张图片")
        return urls

    def _upload_to_minio(self, minio_client: Minio, local_path: str, object_name: str) -> str | None:
        """
        将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
        :param minio_client: 初始化完成的MinIO客户端对象
        :param local_path: 图片本地完整路径
        :param object_name: MinIO中要存储的对象名称
        :return: 图片MinIO访问URL（上传失败返回None）
        """
        try:
            logger.info(f"开始上传图片至MinIO：本地路径={local_path}，MinIO对象名={object_name}")
            # 上传本地文件至MinIO（fput_object：文件流上传，适合大文件）
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
            )

            # 处理路径特殊字符，避免URL解析错误
            object_name = object_name.replace("\\", "%5C")

            # 根据配置选择HTTP/HTTPS协议
            protocol = "https" if minio_config.minio_secure else "http"

            # 构造MinIO基础访问URL
            base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"

            # 拼接完整图片访问URL
            img_url = f"{base_url}/{object_name}"
            logger.info(f"图片上传成功，访问URL：{img_url}")

            return img_url
        except Exception as e:
            logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")
            return None

    def _merge_summary_and_url(self, summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
        :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
        """

        image_info = {}

        # 遍历摘要字典，仅保留有对应URL的图片
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)

        logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")
        return image_info

    def _process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """
        核心功能：替换MD内容中的本地图片引用为MinIO远程引用
        替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
        :param md_content: 原始MD文件内容
        :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
        :return: 替换后的新MD内容
        """

        # 遍历 image_info 字典的每一项：key=图片文件名，value=(摘要, 新URL)
        for image_file, (summary, new_url) in image_info.items():
            # 正则匹配MD图片标签，忽略大小写
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

            # 替换匹配内容：使用新摘要作为图片描述，新URL作为图片路径
            md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
            logger.info(f"完成MD图片引用替换：{image_file} → {new_url}")

        logger.info(f"MD文件图片引用替换完成，共替换{len(image_info)}处图片引用")

        return md_content

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        # 构造新文件路径：替换原后缀为 _new.md
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        # 写入新MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")

        return new_md_file_name


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    # 测试MD文件路径
    md_path = r"D:\opencode\knowledge_base\output\hak180产品安全手册\hak180产品安全手册.md"

    # 构造测试状态对象，模拟流程入参
    init_state = create_default_state(
        task_id="task_001",
        md_path=md_path,
        md_content=""
    )

    # 执行核心处理流程
    node_md_img = NodeMdImg()
    final_state = node_md_img(init_state)

    print(f"\n=== 图片处理完成 ===")
    print(f"新md_path: {final_state['md_path']}")
    print(f"md_content 长度: {len(final_state['md_content'])} 字符")
