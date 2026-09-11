from functools import lru_cache

from app.core.embedding_config import embedding_config
from app.core.logger import logger


@lru_cache(maxsize=1)
def get_bge_m3_ef():
    """
    获取 BGE-M3 嵌入模型单例（懒加载，复用模型实例）
    支持稠密向量（dense）+ 稀疏向量（sparse）混合输出
    """
    from FlagEmbedding import BGEM3FlagModel

    logger.info(f"开始加载 BGE-M3 模型，路径：{embedding_config.bge_m3_path}，设备：{embedding_config.bge_device}")
    model = BGEM3FlagModel(
        embedding_config.bge_m3_path or embedding_config.bge_m3,
        use_fp16=embedding_config.bge_fp16,
        device=embedding_config.bge_device,
    )
    logger.info("BGE-M3 模型加载完成")
    return model


def close_bge_m3_ef():
    """
    关闭并释放 BGE-M3 模型占用的内存/显存
    - 清除 lru_cache 缓存，让模型对象可被垃圾回收
    - 释放 GPU 显存（如果用了 GPU）
    """
    import gc

    get_bge_m3_ef.cache_clear()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()
    logger.info("BGE-M3 模型已关闭并释放内存")


def generate_embeddings(texts: list) -> dict:
    """
    为文本列表批量生成 BGE-M3 稠密 + 稀疏向量
    :param texts: 文本列表
    :return: {"dense": [稠密向量列表], "sparse": [稀疏向量字典列表]}
    """
    model = get_bge_m3_ef()

    # 新版 API：encode 返回 dense_vecs + lexical_weights
    embeddings = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_list = [emb.tolist() for emb in embeddings["dense_vecs"]]
    sparse_list = embeddings["lexical_weights"]

    # 稀疏向量字典里的权重是 np.float32，转成 Python 原生 float 适配 Milvus 存储
    sparse_list = [{int(k): float(v) for k, v in sparse_dict.items()} for sparse_dict in sparse_list]

    return {
        "dense": dense_list,
        "sparse": sparse_list,
    }
