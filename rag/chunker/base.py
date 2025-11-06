"""
Chunker基类模块

定义所有Chunker的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Any, Iterator


class ChunkerBase(ABC):
    """
    Chunker抽象基类

    所有文件分块器必须继承此类并实现chunker方法
    """

    def __init__(self, filepath: str, issemantic: bool = False) -> None:
        """
        初始化Chunker

        参数:
            filepath: 文件路径
            issemantic: 是否使用语义分块(默认False)
        """
        self.filepath = filepath
        self.issemantic = issemantic

    @abstractmethod
    def chunker(self) -> Iterator[Any]:
        """
        分块方法 - 子类必须实现

        返回:
            文档块的迭代器
        """
        raise NotImplementedError(f"{self.__class__.__name__} 必须实现 chunker 方法")
