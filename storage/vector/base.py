"""
向量存储抽象基类

定义统一的向量数据库接口,支持多种向量数据库实现
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """文档数据类"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """搜索结果数据类"""
    document: Document
    score: float
    distance: float


class VectorStoreBase(ABC):
    """
    向量存储抽象基类

    所有向量数据库实现必须继承此类
    """

    def __init__(self, collection_name: str, embedding_dim: int = 1024):
        """
        初始化向量存储

        参数:
            collection_name: 集合/表名
            embedding_dim: 向量维度(默认1024,对应bge-large-zh-v1.5)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    @abstractmethod
    def create_collection(self, drop_if_exists: bool = False) -> None:
        """
        创建集合/表

        参数:
            drop_if_exists: 如果已存在是否删除重建
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, documents: List[Document]) -> List[str]:
        """
        插入文档

        参数:
            documents: 文档列表(必须包含embedding)

        返回:
            插入的文档ID列表
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        向量相似度搜索

        参数:
            query_embedding: 查询向量
            top_k: 返回top-k个结果
            filter_expr: 过滤表达式(可选)

        返回:
            搜索结果列表
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        删除文档

        参数:
            ids: 文档ID列表

        返回:
            删除的文档数量
        """
        raise NotImplementedError

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据ID获取文档

        参数:
            ids: 文档ID列表

        返回:
            文档列表
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        获取文档总数

        返回:
            文档数量
        """
        raise NotImplementedError

    @abstractmethod
    def drop_collection(self) -> None:
        """删除集合/表"""
        raise NotImplementedError

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        pass
