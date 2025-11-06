"""
统一存储管理器 - 管理所有数据库连接
"""
from typing import Optional
from .vector import MilvusStore
from config import get_config


class StorageManager:
    """
    统一存储管理器

    职责：
    1. 管理 Milvus、Elasticsearch、PostgreSQL 连接
    2. 提供统一的初始化接口
    3. 资源生命周期管理
    """

    def __init__(self):
        """初始化存储管理器"""
        self.config = get_config()

        # 向量存储（Milvus）
        self.milvus: Optional[MilvusStore] = None

        # 搜索引擎（Elasticsearch）- 待实现
        # self.es: Optional[ElasticsearchStore] = None

        # 关系数据库（PostgreSQL）- 待实现
        # self.db: Optional[PostgreSQLDatabase] = None

    def initialize_milvus(
        self,
        collection_name: str = "rag_documents",
        embedding_dim: int = 1024
    ) -> MilvusStore:
        """
        初始化 Milvus 连接

        参数:
            collection_name: 集合名称
            embedding_dim: 向量维度

        返回:
            MilvusStore 实例
        """
        if self.milvus is None:
            self.milvus = MilvusStore(
                collection_name=collection_name,
                embedding_dim=embedding_dim
            )
        return self.milvus

    def get_milvus(self) -> MilvusStore:
        """获取 Milvus 实例"""
        if self.milvus is None:
            raise RuntimeError("Milvus 未初始化，请先调用 initialize_milvus()")
        return self.milvus

    def close(self):
        """关闭所有连接"""
        if self.milvus:
            self.milvus.__exit__(None, None, None)
