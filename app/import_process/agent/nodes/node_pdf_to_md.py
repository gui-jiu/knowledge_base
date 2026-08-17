import shutil
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger
from app.core.mineru_config import mineru_config
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


class NodePdfToMd(NodeBase):
    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_pdf_to_md"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        必要参数：task_id、pdf_path、local_dir
        更新参数：md_path、md_content
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # 步骤3：下载ZIP包并提取MD文件
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 步骤4：读取md的内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 步骤5：更新state状态
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """

        # 1、参数非空校验
        pdf_path = state.get("pdf_path", "").strip()
        local_dir = state.get("local_dir", "").strip()
        if not pdf_path:
            raise ValueError("缺失参数：pdf_path")
        if not local_dir:
            raise ValueError("缺失参数：local_dir")

        # 2、转换为Path对象统一处理路径
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(local_dir)

        # 3、PDF文件有效性校验
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")

        # 4、确保输出目录存在，不存在则递归创建
        if not output_dir_obj.exists():
            logger.info(f"输出目录不存在，自动创建：{output_dir_obj.absolute()}")
            output_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1、参数校验
        if not mineru_config.base_url or not mineru_config.api_token:
            raise ValueError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 和 MINERU_BASE_URL 参数")
        logger.info(f"【配置校验】MinerU配置校验成功，开始处理文件：{pdf_path_obj.name}")

        # 2、从MinerU服务器获取上传链接
        token = mineru_config.api_token
        url = f"{mineru_config.base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        logger.info(f"【获取上传链接】调用接口：{url}，请求参数：{data}")

        # 调用接口：获取上传url和任务的batch_id
        response = requests.post(url, headers=header, json=data)

        # 对响应结果进行校验
        # 先校验http状态
        if response.status_code != 200:
            raise RuntimeError(f"【获取上传链接】响应失败：状态码：{response.status_code}，响应结果：{response}")
        # 再校验业务码
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"【获取上传链接】接口调用业务失败：返回数据：{result}")

        # 获取响应结果
        signed_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"]
        logger.info(f"【获取上传链接】成功：上传链接已生成，batch_id：{batch_id}")

        # 3、文件上传
        logger.info(f"【文件上传】开始上传PDF文件：{pdf_path_obj.name}")
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(signed_url, data=f)
            if res_upload.status_code != 200:
                raise RuntimeError(f"【文件上传】上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")
            print(f"【文件上传】成功！")

        # 4、轮询解析结果
        poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

        start_time = time.time()  # 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔时间
        logger.info(f"【任务轮询】开始轮询解析结果，最大超时：{timeout_seconds}s，batch_id：{batch_id}")

        # 根据batch_id轮询任务状态直到成功"done"
        while True:

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            # 发起轮询请求，短超时10秒，异常则重试
            try:
                res_poll = requests.get(url=poll_url, headers=header, timeout=10)
            except Exception as e:
                logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}")
                time.sleep(poll_interval)
                continue

            # 处理HTTP响应错误
            if res_poll.status_code != 200:
                raise RuntimeError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            # 解析轮询结果，校验业务状态
            poll_data = res_poll.json()
            if poll_data["code"] != 0:
                raise RuntimeError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]

            # 获取结果
            result_item = extract_results[0]
            data_state = result_item["state"]

            # 状态为 done
            if data_state == "done":
                logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，bactch_id：{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}")

                return full_zip_url

            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise RuntimeError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                logger.info(
                    f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
        """

        # 1、下载ZIP包
        logger.info(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")
        response = requests.get(zip_url)

        # 对响应结果进行校验
        if response.status_code != 200:
            raise RuntimeError(f"【ZIP下载】ZIP包下载失败：状态码：{response.status_code}，响应结果：{response}")

        # 拼接ZIP包保存路径并保存
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}")

        # 2、清空解压目录
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
            logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 3、创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 4、解压
        logger.info(f"【ZIP解压】开始解压ZIP包：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        # 5、重命名
        logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    # 组装文件路径（直接使用绝对路径）
    pdf_path = r"D:\opencode\资料\hak180产品安全手册.pdf"
    # 组装输出路径
    local_dir = r"D:\opencode\knowledge_base\output"

    # 当前节点图状态初始值
    init_state = create_default_state(
        task_id="task_001",
        pdf_path=pdf_path,
        local_dir=local_dir
    )

    # 执行节点的业务调用
    node_pdf_to_md = NodePdfToMd()
    final_state = node_pdf_to_md(init_state)

    print(f"\n=== 转换完成 ===")
    print(f"md_path: {final_state['md_path']}")
    print(f"md_content 长度: {len(final_state['md_content'])} 字符")
