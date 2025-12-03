"""
Forum State - 论坛讨论共享状态
三个Agent（Moderator、RAG Critic、Web Critic）共享此状态
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import add_messages


class ForumState(TypedDict):
    """
    Forum讨论的共享状态

    三个Agent通过此状态进行信息传递和协作
    """

    # ==================== 会话信息 ====================
    session_id: str  # 会话ID
    user_id: str  # 用户ID

    # ==================== 讨论主题 ====================
    message: str  # 历史信息包括AI提问和用户回答
    interview_context: Optional[Dict[str, Any]]  # 面试上下文（JD、简历等）

    # ==================== 讨论进度控制 ====================
    current_round: int  # 当前讨论轮次（从1开始）
    max_rounds: int  # 最大讨论轮次限制
    current_speaker: str  # 当前发言者标识

    # ==================== 各Agent的评论（当前轮） ====================
    rag_critic_comment: Optional[Dict[str, Any]]  # RAG Critic的评论
    web_critic_comment: Optional[Dict[str, Any]]  # Web Critic的评论

    # ==================== 讨论历史（所有轮次） ====================
    messages: Annotated[List[Dict[str, str]], add_messages]  # LangChain消息历史
    discussion_history: List[Dict[str, Any]]  # 结构化讨论记录
    # 格式：[
    #   {
    #     "round": 1,
    #     "agent": "rag_critic",
    #     "comment": {...},
    #     "timestamp": "2025-12-01 10:00:00"
    #   },
    #   ...
    # ]

    # ==================== 最终评价（由Moderator生成） ====================
    final_evaluation: Optional[Dict[str, Any]]  # 最终评价
    # 格式：{
    #   "overall_score": 85,
    #   "dimensions": {
    #     "completeness": 80,
    #     "accuracy": 90,
    #     "depth": 85
    #   },
    #   "strengths": ["要点1", "要点2"],
    #   "improvements": ["建议1", "建议2"],
    #   "summary": "总结文字"
    # }

    # ==================== 控制信号 ====================
    next_step: str  # 下一步动作
    # 可能值：
    # - "rag_critic": RAG Critic发言
    # - "web_critic": Web Critic发言
    # - "moderator_decide": Moderator决定是否继续
    # - "moderator_summarize": Moderator汇总评价
    # - "save": 保存到数据库
    # - "end": 结束讨论

    should_continue: bool  # Moderator的决策：是否继续下一轮

    # ==================== 元数据 ====================
    metadata: Optional[Dict[str, Any]]  # 其他元数据（token使用、耗时等）
