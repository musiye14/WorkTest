"""
RAG Critic Agent 图构建
"""

from langgraph.graph import StateGraph, END
from .state import RAGCriticState
from .nodes import RAGCriticNodes
from storage.manager import StorageManager
from storage.database.postgresql import PostgreSQLDatabase
from rag.embedding import YEmbedding


def buildGraph(
    llm,
    storage_manager: StorageManager,
    db: PostgreSQLDatabase,
    embedding: YEmbedding,
    top_k: int = 3
):
    """
    构建 RAG Critic Agent 图

    Args:
        llm: LLM 实例
        storage_manager: 存储管理器
        db: PostgreSQL 数据库实例
        embedding: 嵌入模型实例
        top_k: 检索相似案例的数量（默认 3）

    Returns:
        编译后的图
    """
    nodes = RAGCriticNodes(
        llm=llm,
        storage_manager=storage_manager,
        db=db,
        embedding=embedding,
        top_k=top_k
    )

    builder = StateGraph(RAGCriticState)

    # 添加节点
    builder.add_node("search_similar_cases", nodes.search_similar_cases)
    builder.add_node("generate_comment", nodes.generate_comment)

    # 添加边
    builder.add_edge("search_similar_cases", "generate_comment")
    builder.add_edge("generate_comment", END)

    # 设置入口点
    builder.set_entry_point("search_similar_cases")

    return builder.compile()
