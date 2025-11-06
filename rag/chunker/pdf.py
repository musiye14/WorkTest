"""
PDF文件分块器
"""
from typing import Any, Iterator
from langchain_community.document_loaders import PyPDFLoader
from .base import ChunkerBase
from .registry import ChunkerRegistry


@ChunkerRegistry.register('pdf')
class PDFChunk(ChunkerBase):
    """PDF文件分块器 - 按页面分块"""

    def chunker(self) -> Iterator[Any]:
        """
        加载PDF文件并按页面分块

        返回:
            Document对象的迭代器,每个Document代表一页
        """
        loader = PyPDFLoader(self.filepath)
        pages = loader.load()
        return iter(pages)