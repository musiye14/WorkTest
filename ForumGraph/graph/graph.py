"""
Forum Graph 图构建
协调 RAGCriticAgent、WebCriticAgent、ModeratorAgent
"""

from langgraph.graph import StateGraph, END
from .state import ForumState
from .nodes import ForumNodes
from RAGCriticAgent import RAGCriticAgent
from WebCriticAgent import WebCriticAgent
from ModeratorAgent import ModeratorAgent


def buildGraph(
    rag_critic_agent: RAGCriticAgent,
    web_critic_agent: WebCriticAgent,
    moderator_agent: ModeratorAgent
):
    """
    构建 Forum Agent 图

    Args:
        rag_critic_agent: RAG Critic Agent 实例
        web_critic_agent: Web Critic Agent 实例
        moderator_agent: Moderator Agent 实例

    Returns:
        编译后的图
    """
    nodes = ForumNodes(
        rag_critic_agent=rag_critic_agent,
        web_critic_agent=web_critic_agent,
        moderator_agent=moderator_agent
    )

    builder = StateGraph(ForumState)

    # 添加节点
    builder.add_node("rag_critic", nodes.rag_critic_node)
    builder.add_node("web_critic", nodes.web_critic_node)
    builder.add_node("moderator_decide", nodes.moderator_decide_node)
    builder.add_node("moderator_summarize", nodes.moderator_summarize_node)
    builder.add_node("save", nodes.save_discussion_node)

    # 添加边
    # RAG Critic → Web Critic
    builder.add_edge("rag_critic", "web_critic")

    # Web Critic → Moderator Decide
    builder.add_edge("web_critic", "moderator_decide")

    # Moderator Decide → 条件边（根据 should_continue 决定）
    builder.add_conditional_edges(
        "moderator_decide",
        nodes.decide_next_step,
        {
            "rag_critic": "rag_critic",  # 继续下一轮
            "moderator_summarize": "moderator_summarize",  # 结束讨论，生成最终评价
            "end": END
        }
    )

    # Moderator Summarize → Save
    builder.add_edge("moderator_summarize", "save")

    # Save → END
    builder.add_edge("save", END)

    # 设置入口点
    builder.set_entry_point("rag_critic")

    return builder.compile()
