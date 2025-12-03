"""
Web Critic Agent 状态定义
"""

from typing import TypedDict, Optional, Dict, Any, List


class WebCriticState(TypedDict):
    """Web Critic Agent 的内部状态"""

    # 输入信息
    question: str  # 面试问题
    user_answer: str  # 用户回答

    # 搜索结果
    search_results: List[Dict[str, Any]]  # 从 Tavily API 搜索到的结果

    # 评论结果
    web_comment: Optional[Dict[str, Any]]  # 生成的评论

    # 元数据
    session_id: str
    user_id: str
    metadata: Optional[Dict[str, Any]]
