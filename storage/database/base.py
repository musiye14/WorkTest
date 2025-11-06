"""
数据库抽象基类 - PostgreSQL
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class DatabaseBase(ABC):
    """
    数据库抽象基类

    用于存储完整的原始数据
    """

    @abstractmethod
    async def connect(self) -> None:
        """连接数据库"""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        raise NotImplementedError

    @abstractmethod
    async def insert_document(self, document: Dict[str, Any]) -> str:
        """
        插入文档

        参数:
            document: 文档数据

        返回:
            文档 ID
        """
        raise NotImplementedError

    @abstractmethod
    async def insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        批量插入文档

        参数:
            documents: 文档列表

        返回:
            文档 ID 列表
        """
        raise NotImplementedError

    @abstractmethod
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取文档

        参数:
            doc_id: 文档 ID

        返回:
            文档数据
        """
        raise NotImplementedError

    @abstractmethod
    async def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量获取文档

        参数:
            doc_ids: 文档 ID 列表

        返回:
            文档列表
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """
        删除文档

        参数:
            doc_id: 文档 ID

        返回:
            是否成功
        """
        raise NotImplementedError

    @abstractmethod
    async def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新文档

        参数:
            doc_id: 文档 ID
            updates: 更新内容

        返回:
            是否成功
        """
        raise NotImplementedError
