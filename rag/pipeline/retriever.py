"""
检索器 - RAG检索功能
"""
from typing import List, Optional
from storage.vector import VectorStoreBase, SearchResult
from ..embedding import YEmbedding


class Retriever:
    """
    检索器 - 向量相似度检索

    功能:
        1. 将查询文本转换为向量
        2. 在向量数据库中搜索相似文档
        3. 返回相关文档
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_model: Optional[YEmbedding] = None
    ):
        """
        初始化检索器

        参数:
            vector_store: 向量存储实例
            embedding_model: 嵌入模型(可选)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or YEmbedding()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        检索相关文档

        参数:
            query: 查询文本
            top_k: 返回top-k个结果
            filter_expr: 过滤表达式(可选)

        返回:
            搜索结果列表
        """
        # 1. 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)

        # 2. 向量搜索
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr
        )

        return results

    def format_results(self, results: List[SearchResult]) -> str:
        """
        格式化搜索结果为可读文本

        参数:
            results: 搜索结果列表

        返回:
            格式化后的文本
        """
        if not results:
            return "未找到相关文档"

        formatted = []
        for i, result in enumerate(results, 1):
            doc = result.document
            formatted.append(
                f"[{i}] (相似度: {result.score:.4f})\n"
                f"文件: {doc.metadata.get('filename', 'Unknown')}\n"
                f"内容: {doc.content[:200]}...\n"
            )

        return "\n".join(formatted)
