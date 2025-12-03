"""
Forum Graph 节点定义
负责协调 RAGCriticAgent、WebCriticAgent、ModeratorAgent
"""

from .state import ForumState
from typing import Dict, Any
from RAGCriticAgent import RAGCriticAgent
from WebCriticAgent import WebCriticAgent
from ModeratorAgent import ModeratorAgent


class ForumNodes:
    """Forum 节点类，封装所有节点逻辑"""

    def __init__(
        self,
        rag_critic_agent: RAGCriticAgent,
        web_critic_agent: WebCriticAgent,
        moderator_agent: ModeratorAgent,
        agent_name: str = "ForumAgent"
    ):
        """
        初始化 Forum 节点

        Args:
            rag_critic_agent: RAG Critic Agent 实例
            web_critic_agent: Web Critic Agent 实例
            moderator_agent: Moderator Agent 实例
            agent_name: Agent 名称，用于日志记录
        """
        self.rag_critic = rag_critic_agent
        self.web_critic = web_critic_agent
        self.moderator = moderator_agent
        self.agent_name = agent_name

    async def rag_critic_node(self, state: ForumState) -> Dict[str, Any]:
        """
        RAG Critic 节点

        调用 RAGCriticAgent 生成基于历史面经的评论
        """
        # 直接传递 state 给 RAG Critic Agent
        updated_state = await self.rag_critic.run(state)

        # 返回更新后的字段
        return {
            "rag_critic_comment": updated_state.get("rag_critic_comment"),
            "current_speaker": "rag_critic"
        }

    async def web_critic_node(self, state: ForumState) -> Dict[str, Any]:
        """
        Web Critic 节点

        调用 WebCriticAgent 生成基于网络搜索的评论
        """
        # 直接传递 state 给 Web Critic Agent
        updated_state = await self.web_critic.run(state)

        # 返回更新后的字段
        return {
            "web_critic_comment": updated_state.get("web_critic_comment"),
            "current_speaker": "web_critic"
        }

    async def moderator_decide_node(self, state: ForumState) -> Dict[str, Any]:
        """
        Moderator 决策节点

        决定是否继续下一轮讨论
        """
        result = await self.moderator.decide_next_step(state)
        return result

    async def moderator_summarize_node(self, state: ForumState) -> Dict[str, Any]:
        """
        Moderator 总结节点

        生成最终评价
        """
        result = await self.moderator.generate_final_evaluation(state)
        return result

    async def save_discussion_node(self, state: ForumState) -> Dict[str, Any]:
        """
        保存讨论节点

        将讨论结果保存到数据库
        """
        from storage.database.postgresql import PostgreSQLDatabase
        from config import get_config

        # 从state中提取信息
        message = state.get("message", "")
        question, user_answer = self._parse_message(message)

        # 构造讨论记录
        discussion = {
            'session_id': state.get('session_id', ''),
            'user_id': state.get('user_id', ''),
            'question': question or '',
            'user_answer': user_answer or '',
            'rag_comment': state.get('rag_critic_comment'),
            'web_comment': state.get('web_critic_comment'),
            'final_evaluation': state.get('final_evaluation'),
            'discussion_history': state.get('discussion_history', []),
            'total_rounds': state.get('current_round', 1),
            'metadata': {
                'interview_context': state.get('interview_context'),
                'max_rounds': state.get('max_rounds', 3)
            }
        }

        # 保存到数据库
        config = get_config()
        db_url = (
            f"postgresql://{config.get('POSTGRES_USER')}:{config.get('POSTGRES_PASSWORD')}@"
            f"{config.get('POSTGRES_HOST')}:{config.get('POSTGRES_PORT')}/{config.get('POSTGRES_DB')}"
        )
        db = PostgreSQLDatabase(db_url)
        await db.connect()

        try:
            discussion_id = await db.insert_forum_discussion(discussion)
            print(f"[OK] Forum讨论已保存，ID: {discussion_id}")

            return {
                "discussion_id": discussion_id,
                "next_step": "end"
            }
        finally:
            await db.close()

    def decide_next_step(self, state: ForumState) -> str:
        """
        条件边：决定下一步去哪个节点

        Returns:
            下一个节点的名称
        """
        next_step = state.get("next_step", "end")

        # 根据 next_step 决定路由
        if next_step == "rag_critic":
            return "rag_critic"
        elif next_step == "web_critic":
            return "web_critic"
        elif next_step == "moderator_decide":
            return "moderator_decide"
        elif next_step == "moderator_summarize":
            return "moderator_summarize"
        elif next_step == "save":
            return "save"
        else:
            return "end"

    def _parse_message(self, message: str) -> tuple:
        """
        从 message 中提取问题和用户回答

        Args:
            message: 历史消息字符串

        Returns:
            (question, user_answer) 元组
        """
        lines = message.strip().split('\n')

        question = None
        user_answer = None

        for line in lines:
            line = line.strip()
            if line.startswith("面试官：") or line.startswith("AI："):
                question = line.split("：", 1)[1].strip()
            elif line.startswith("用户：") or line.startswith("候选人："):
                user_answer = line.split("：", 1)[1].strip()

        return question, user_answer
