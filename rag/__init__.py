"""
RAG模块 - 文档分块、嵌入和检索

使用示例:
    from rag import RAG
    from storage import StorageManager

    # 1. 初始化存储管理器
    storage = StorageManager()
    storage.initialize_milvus("my_docs")

    # 2. 初始化 RAG
    rag = RAG(storage)

    # 3. 添加文档
    rag.add_documents(documents)

    # 4. 检索
    results = rag.search("查询内容", top_k=5)
"""
# 延迟导入，避免加载不需要的依赖（torch等）
from .chunker import ChunkerRegistry, ChunkerBase
from .chunker import PDFChunk, TxtChunker  # 按需导入，避免加载torch
from .pipeline import DocumentProcessor, Retriever  # 按需导入
from .embedding import YEmbedding  # 按需导入
from .rag import RAG  # 按需导入
from .reranker import Reranker  # 按需导入

__all__ = [
    'ChunkerRegistry',
    'ChunkerBase',
    'PDFChunk',
    'TxtChunker',
    'DocumentProcessor',
    'Retriever',
    'YEmbedding',
    'RAG',
    'Reranker'
]
