"""
重排序模块 - 对检索结果进行重排序
"""
from typing import List
from storage.vector.base import SearchResult


class Reranker:
    """
    重排序器

    功能：
    1. 对混合检索结果进行重排序
    2. 提升检索精度

    TODO: 集成 bge-reranker-large 模型
    """

    def __init__(self, model_path: str = None):
        """
        初始化重排序器

        参数:
            model_path: 重排序模型路径（可选）
        """
        self.model_path = model_path
        # TODO: 加载重排序模型
        # self.model = load_reranker_model(model_path)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        重排序

        参数:
            query: 查询文本
            results: 初步检索结果
            top_k: 返回 top-k 个结果

        返回:
            重排序后的结果
        """
        # TODO: 实现重排序逻辑
        # 1. 使用 reranker 模型计算 query 和每个 doc 的相关性分数
        # 2. 按新分数排序
        # 3. 返回 top_k

        # 当前简单实现：按原始分数排序
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
