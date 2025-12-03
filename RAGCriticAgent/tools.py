"""
RAG Critic Agent 工具集
负责从 episodic_memory 检索相似面经案例
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from storage.manager import StorageManager
from storage.database.postgresql import PostgreSQLDatabase
from rag.embedding import YEmbedding


# 全局存储管理器实例（需要在使用前初始化）
_storage_manager: Optional[StorageManager] = None
_db: Optional[PostgreSQLDatabase] = None
_embedding: Optional[YEmbedding] = None


def initialize_tools(storage_manager: StorageManager, db: PostgreSQLDatabase, embedding: YEmbedding):
    """初始化工具（在使用前必须调用）"""
    global _storage_manager, _db, _embedding
    _storage_manager = storage_manager
    _db = db
    _embedding = embedding


@tool
async def search_similar_interview_cases(
    question: str,
    company: Optional[str] = None,
    difficulty: Optional[str] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    从 episodic_memory 检索相似的面试案例

    使用场景：
    - 查找相似的面试问题和标准答案
    - 对比用户回答与历史优秀答案
    - 获取该问题的评分标准和关键点

    参数：
        question: 面试问题文本
        company: 目标公司（可选，如 "字节跳动"）
        difficulty: 难度级别（可选，"简单"/"中等"/"困难"）
        top_k: 返回数量（默认 3）

    返回：
        情节记忆列表，包含相似问题、标准答案、评分要点
    """
    if not _storage_manager or not _db or not _embedding:
        raise RuntimeError("工具未初始化，请先调用 initialize_tools()")

    # 1. 将问题转换为 embedding
    query_embedding = _embedding.embed_query(question)

    # 2. 在 Milvus episodic_memory_vectors 中检索
    milvus = _storage_manager.get_milvus()

    # 构建过滤表达式
    filter_conditions = []

    # 过滤公司
    if company:
        filter_conditions.append(f'company == "{company}"')

    # 过滤难度
    if difficulty:
        filter_conditions.append(f'difficulty == "{difficulty}"')

    # 过滤质量评分（只返回高质量案例）
    filter_conditions.append('quality_score >= 7')

    # 组合过滤条件
    filter_expr = ' and '.join(filter_conditions) if filter_conditions else None

    search_results = milvus.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter_expr=filter_expr
    )

    # 3. 获取 doc_ids
    if not search_results:
        return []

    doc_ids = [result.document.id for result in search_results]

    # 4. 从 PostgreSQL 批量查询完整数据
    memories = await _db.get_episodic_memory_by_ids(doc_ids)

    # 5. 按照相似度排序返回（保持 Milvus 的排序）
    id_to_memory = {str(m['id']): m for m in memories}
    sorted_memories = [id_to_memory[doc_id] for doc_id in doc_ids if doc_id in id_to_memory]

    return sorted_memories


# 导出所有工具
RAG_CRITIC_TOOLS = [
    search_similar_interview_cases
]


def get_rag_critic_tools() -> List:
    """获取 RAG Critic Agent 的所有工具"""
    return RAG_CRITIC_TOOLS
