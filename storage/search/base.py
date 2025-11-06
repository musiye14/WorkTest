"""
搜索引擎抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..vector.base import Document, SearchResult


class SearchEngineBase(ABC):
    """
    搜索引擎抽象基类

    用于 BM25 全文检索
    """

    def __init__(self, index_name: str):
        """
        初始化搜索引擎

        参数:
            index_name: 索引名称
        """
        self.index_name = index_name

    @abstractmethod
    def create_index(self, drop_if_exists: bool = False) -> None:
        """
        创建索引

        参数:
            drop_if_exists: 如果已存在是否删除重建
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, documents: List[Document]) -> List[str]:
        """
        插入文档

        参数:
            documents: 文档列表（只需 id 和 content）

        返回:
            插入的文档 ID 列表
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        BM25 全文检索

        参数:
            query: 查询文本
            top_k: 返回 top-k 个结果
            filters: 过滤条件

        返回:
            搜索结果列表（只包含 doc_id 和 score）
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        删除文档

        参数:
            ids: 文档 ID 列表

        返回:
            删除的文档数量
        """
        raise NotImplementedError

    @abstractmethod
    def drop_index(self) -> None:
        """删除索引"""
        raise NotImplementedError
