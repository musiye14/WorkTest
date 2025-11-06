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
from .chunker import ChunkerRegistry, ChunkerBase, PDFChunk, TxtChunker
from .pipeline import DocumentProcessor, Retriever
from .embedding import YEmbedding
from .rag import RAG
from .reranker import Reranker

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
