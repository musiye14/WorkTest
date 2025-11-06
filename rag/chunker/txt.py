"""
文本文件分块器 - 支持txt、md等纯文本格式
"""
from typing import Any, Iterator, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from ..embedding import YEmbedding
from .base import ChunkerBase
from .registry import ChunkerRegistry


@ChunkerRegistry.register('txt', 'md', 'text')
class TxtChunker(ChunkerBase):
    """文本文件分块器 - 支持字符分块和语义分块"""

    def chunker(self) -> Iterator[Any]:
        """
        加载文本文件并分块

        根据issemantic参数选择分块策略:
        - False: 使用RecursiveCharacterTextSplitter(字符分块)
        - True: 使用SemanticChunker(语义分块)

        返回:
            文本块的迭代器
        """
        if not self.issemantic:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "。", "?", "？", "!", "！"]
            )
        else:
            model = YEmbedding()
            splitter = SemanticChunker(model, breakpoint_threshold_type="standard_deviation")

        with open(self.filepath, mode="r", encoding="utf-8") as f:
            data = f.read()
            data = data.replace("\n\n", "\n")
            docs = splitter.split_text(data)

        print(f"总共分割了{len(docs)}个chunk")
        return iter(docs)


            

