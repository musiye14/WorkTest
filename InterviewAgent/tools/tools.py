"""
面试官 Agent 工具集
使用 langchain_core.tools 注册
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from storage.manager import StorageManager
from storage.database.postgresql import PostgreSQLDatabase
from embedding import YEmbedding


# 全局存储管理器实例（需要在使用前初始化）
_storage_manager: Optional[StorageManager] = None
_db: Optional[PostgreSQLDatabase] = None
_embeding: Optional[YEmbedding] = None


def initialize_tools(storage_manager: StorageManager, db: PostgreSQLDatabase, embeding: YEmbedding):
    """初始化工具（在使用前必须调用）"""
    global _storage_manager, _db, _embeding
    _storage_manager = storage_manager
    _db = db
    _embeding = embeding


@tool
async def search_semantic_memory(
    user_id: str,
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    查询用户的语义记忆（知识点掌握度、历史表现）

    使用场景：
    - 训练模式下，需要针对薄弱点提问
    - 需要对比用户的进步情况
    - 查询用户在某个知识点上的历史表现

    参数：
        user_id: 用户 ID
        query: 查询内容（知识点名称或描述）
        top_k: 返回数量（默认 5）

    返回：
        语义记忆列表，包含知识点掌握度、练习次数、薄弱点等信息
    """
    if not _storage_manager or not _db or not _embeding:
        raise RuntimeError("工具未初始化，请先调用 initialize_tools()")

    # 1. 将 query 转换为 embedding
    query_embedding = _embeding.embed_query(query)

    # 2. 在 Milvus semantic_memory_vectors 中检索
    milvus = _storage_manager.get_milvus()

    # 构建过滤表达式（只查询该用户的记忆）
    filter_expr = f'user_id == "{user_id}"'

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
    memories = await _db.get_semantic_memory_by_ids(doc_ids)

    # 5. 按照相似度排序返回（保持 Milvus 的排序）
    id_to_memory = {str(m['id']): m for m in memories}
    sorted_memories = [id_to_memory[doc_id] for doc_id in doc_ids if doc_id in id_to_memory]

    return sorted_memories


@tool
async def search_episodic_memory(
    query: str,
    company: Optional[str] = None,
    difficulty: Optional[str] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    查询情节记忆（Few-shot 案例库，相似面试问题）

    使用场景：
    - 不确定如何提问某个技术点
    - 需要了解行业标准问法
    - 参考真实面经案例

    参数：
        query: 查询内容（技术点或问题描述）
        company: 目标公司（可选，如 "字节跳动"）
        difficulty: 难度级别（可选，"简单"/"中等"/"困难"）
        top_k: 返回数量（默认 3）

    返回：
        情节记忆列表，包含相似的面试问题案例、标准问法、公司风格
    """
    if not _storage_manager or not _db or not _embeding:
        raise RuntimeError("工具未初始化，请先调用 initialize_tools()")

    # 1. 将 query 转换为 embedding
    query_embedding = _embeding.embed_query(query)

    # 2. 在 Milvus episodic_memory_vectors 中检索（带过滤条件）
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
INTERVIEWER_TOOLS = [
    search_semantic_memory,
    search_episodic_memory
]


def get_interviewer_tools() -> List:
    """获取面试官 Agent 的所有工具"""
    return INTERVIEWER_TOOLS