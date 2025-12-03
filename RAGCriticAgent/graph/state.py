"""
RAG Critic Agent 状态定义
"""

from typing import TypedDict, Optional, Dict, Any, List
from typing_extensions import Annotated
from langgraph.graph.message import add_messages


class RAGCriticState(TypedDict):
    """RAG Critic Agent 的内部状态"""

    # 输入信息
    question: str  # 面试问题
    user_answer: str  # 用户回答
    interview_context: Optional[Dict[str, Any]]  # 面试上下文（公司、难度等）

    # 检索结果
    similar_cases: List[Dict[str, Any]]  # 从 episodic_memory 检索到的相似案例

    # 评论结果
    rag_comment: Optional[Dict[str, Any]]  # 生成的评论

    # 元数据
    session_id: str
    user_id: str
    metadata: Optional[Dict[str, Any]]
