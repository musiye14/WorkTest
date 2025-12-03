"""
Forum Agent 主类
负责初始化三个 Critic Agent 并协调它们的工作
"""

import uuid
import time
from typing import Dict, Any, Optional, List
from .graph.graph import buildGraph
from .graph.state import ForumState
from RAGCriticAgent import RAGCriticAgent
from WebCriticAgent import WebCriticAgent
from ModeratorAgent import ModeratorAgent
from storage.manager import StorageManager
from storage.database.postgresql import PostgreSQLDatabase
from rag.embedding import YEmbedding
from config import get_config
from langchain_core.messages import HumanMessage, AIMessage


class ForumAgent:
    """Forum Agent 主类"""

    def __init__(
        self,
        llm,
        storage_manager: StorageManager,
        db: PostgreSQLDatabase,
        embedding: YEmbedding,
        user_id: Optional[str] = None,
        max_rounds: int = 3,
        rag_top_k: int = 3,
        web_top_k: int = 5
    ):
        """
        初始化 Forum Agent

        Args:
            llm: LLM 实例
            storage_manager: 存储管理器
            db: PostgreSQL 数据库实例
            embedding: 嵌入模型实例
            user_id: 用户 ID（可选，默认生成随机 ID）
            max_rounds: 最大讨论轮次（默认 3）
            rag_top_k: RAG Critic 检索数量（默认 3）
            web_top_k: Web Critic 搜索结果数量（默认 5）
        """
        self.llm = llm
        self.storage_manager = storage_manager
        self.db = db
        self.embedding = embedding
        self.user_id = user_id or self._generate_user_id()
        self.max_rounds = max_rounds

        # 从配置读取 Tavily API Key
        config = get_config()
        tavily_api_key = config.get('TAVILY_API_KEY')

        if not tavily_api_key:
            raise ValueError("未找到 TAVILY_API_KEY，请在 env 文件中配置")

        # 初始化 Milvus（RAG Critic 需要使用）
        milvus = self.storage_manager.initialize_milvus(
            collection_name="episodic_memory_vectors",
            embedding_dim=1024
        )
        # 创建或加载集合
        milvus.create_collection(drop_if_exists=False)

        # 初始化三个 Agent
        self.rag_critic = RAGCriticAgent(
            llm=llm,
            storage_manager=storage_manager,
            db=db,
            embedding=embedding,
            top_k=rag_top_k
        )

        self.web_critic = WebCriticAgent(
            llm=llm,
            tavily_api_key=tavily_api_key,
            max_search_results=web_top_k
        )

        self.moderator = ModeratorAgent(llm=llm)

        # 构建图
        self.graph = buildGraph(
            rag_critic_agent=self.rag_critic,
            web_critic_agent=self.web_critic,
            moderator_agent=self.moderator
        )

    @staticmethod
    def _generate_user_id() -> str:
        """生成随机用户 ID"""
        return f"user_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _generate_session_id() -> str:
        """生成随机会话 ID"""
        return f"forum_{uuid.uuid4().hex[:12]}"

    async def run_discussion(
        self,
        question: str,
        user_answer: str,
        interview_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        运行完整的 Forum 讨论流程

        Args:
            question: 面试问题
            user_answer: 用户回答
            interview_context: 面试上下文（公司、难度等）

        Returns:
            包含最终评价的结果
        """
        # 构建初始状态
        session_id = self._generate_session_id()

        # 构建 message（历史消息格式）
        message = f"面试官：{question}\n用户：{user_answer}"

        initial_state: ForumState = {
            "session_id": session_id,
            "user_id": self.user_id,
            "message": message,
            "interview_context": interview_context,
            "current_round": 1,
            "max_rounds": self.max_rounds,
            "current_speaker": "",
            "rag_critic_comment": None,
            "web_critic_comment": None,
            "messages": [],
            "discussion_history": [],
            "final_evaluation": None,
            "next_step": "rag_critic",
            "should_continue": True,
            "metadata": None
        }

        # 运行图
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state

        except Exception as e:
            print(f"Forum 讨论过程中发生错误：{e}")
            import traceback
            traceback.print_exc()
            raise

    async def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        company: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        评价用户回答（便捷方法）

        Args:
            question: 面试问题
            user_answer: 用户回答
            company: 公司名称（可选）
            difficulty: 难度（可选）

        Returns:
            包含最终评价的结果
        """
        interview_context = {}
        if company:
            interview_context["company"] = company
        if difficulty:
            interview_context["difficulty"] = difficulty

        return await self.run_discussion(
            question=question,
            user_answer=user_answer,
            interview_context=interview_context
        )

    async def evaluate_interview_session(
        self,
        messages: List,
        interview_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        评价完整的面试会话（处理 InterviewAgent 的 messages 列表）

        Args:
            messages: InterviewAgent 的消息列表 [AIMessage, HumanMessage, AIMessage, HumanMessage, ...]
            interview_context: 面试上下文（公司、岗位、难度等）

        Returns:
            {
                "session_id": "forum_xxx",
                "qa_evaluations": [
                    {
                        "qa_index": 0,
                        "question": "问题1",
                        "answer": "回答1",
                        "rag_comment": {...},
                        "web_comment": {...},
                        "evaluation": {...}
                    },
                    ...
                ],
                "overall_evaluation": {
                    "overall_score": 7.5,
                    "strengths": [...],
                    "weaknesses": [...],
                    "knowledge_gaps": [...],
                    "performance_trend": "improving",
                    "topic_analysis": {...},
                    "improvement_suggestions": [...],
                    "summary": "..."
                },
                "statistics": {
                    "total_questions": 5,
                    "average_score": 7.5,
                    "total_time": 120.5,
                    "total_tokens": 15000
                }
            }
        """
        print("\n" + "="*80)
        print("Forum Agent - 面试会话评价")
        print("="*80)

        session_start_time = time.time()
        session_id = self._generate_session_id()

        # 1. 提取 QA 对
        print("\n[步骤1] 从 messages 中提取 QA 对...")
        qa_pairs = self._extract_qa_pairs(messages)
        print(f"[OK] 提取到 {len(qa_pairs)} 个 QA 对")

        # 2. 逐个评价每个 QA 对
        print("\n[步骤2] 逐个评价每个 QA 对...")
        qa_evaluations = []
        total_tokens = 0

        for i, qa in enumerate(qa_pairs, 1):
            print(f"\n{'='*80}")
            print(f"评价第 {i}/{len(qa_pairs)} 个 QA 对")
            print(f"{'='*80}")
            print(f"问题: {qa['question'][:60]}...")
            print(f"回答: {qa['answer'][:60]}...")

            qa_start_time = time.time()

            # 调用单条评价
            result = await self.run_discussion(
                question=qa['question'],
                user_answer=qa['answer'],
                interview_context=interview_context
            )

            qa_duration = time.time() - qa_start_time

            # 收集评价结果
            qa_evaluations.append({
                "qa_index": i - 1,
                "question": qa['question'],
                "answer": qa['answer'],
                "rag_comment": result.get('rag_critic_comment'),
                "web_comment": result.get('web_critic_comment'),
                "evaluation": result.get('final_evaluation'),
                "duration": qa_duration
            })

            print(f"\n[OK] 第 {i} 个 QA 评价完成 | 耗时: {qa_duration:.2f}s")

        # 3. 生成总体评价
        print(f"\n{'='*80}")
        print("[步骤3] 生成总体评价...")
        print(f"{'='*80}")

        overall_evaluation = await self.moderator.generate_overall_evaluation(
            qa_evaluations=qa_evaluations,
            interview_context=interview_context or {}
        )

        # 4. 计算统计信息
        session_duration = time.time() - session_start_time
        scores = [
            qa['evaluation'].get('overall_score', 0)
            for qa in qa_evaluations
            if qa.get('evaluation') and isinstance(qa['evaluation'], dict)
        ]
        average_score = sum(scores) / len(scores) if scores else 0

        statistics = {
            "total_questions": len(qa_pairs),
            "average_score": round(average_score, 2),
            "total_time": round(session_duration, 2),
            "total_tokens": total_tokens  # TODO: 累计 token 统计
        }

        print(f"\n{'='*80}")
        print("[完成] 面试会话评价完成")
        print(f"{'='*80}")
        print(f"总问题数: {statistics['total_questions']}")
        print(f"平均得分: {statistics['average_score']}/10")
        print(f"总耗时: {statistics['total_time']:.2f}s")
        print(f"{'='*80}\n")

        # 5. 保存到数据库（可选）
        # TODO: 保存 overall_evaluation 到数据库

        return {
            "session_id": session_id,
            "qa_evaluations": qa_evaluations,
            "overall_evaluation": overall_evaluation,
            "statistics": statistics
        }

    def _extract_qa_pairs(self, messages: List) -> List[Dict[str, str]]:
        """
        从 InterviewAgent 的 messages 列表中提取 QA 对

        Args:
            messages: [AIMessage, HumanMessage, AIMessage, HumanMessage, ...]

        Returns:
            [
                {"question": "问题1", "answer": "回答1"},
                {"question": "问题2", "answer": "回答2"},
                ...
            ]
        """
        qa_pairs = []
        current_question = None

        for msg in messages:
            if isinstance(msg, AIMessage):
                # 面试官的问题
                current_question = msg.content
            elif isinstance(msg, HumanMessage):
                # 候选人的回答
                if current_question:
                    qa_pairs.append({
                        "question": current_question,
                        "answer": msg.content
                    })
                    # 注意：不重置 current_question，因为可能有追问
                    # 追问会继续使用同一个主问题

        return qa_pairs


def main():
    """命令行交互式测试入口"""
    from InterviewAgent.llms.openai_llm import OpenAILLM
    import os
    import asyncio

    # 从环境变量获取 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return

    # 初始化 LLM
    llm = OpenAILLM(
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        base_url=base_url
    )

    # TODO: 初始化 storage_manager、db、embedding
    # 这里需要根据实际情况初始化

    print("=" * 50)
    print("Forum Agent 测试")
    print("=" * 50)
    print()

    # 测试问题
    question = "请介绍一下 Redis 的持久化机制？"
    user_answer = "Redis 有两种持久化方式：RDB 和 AOF。RDB 是快照方式，定期保存数据。AOF 是追加日志方式，记录每个写操作。"

    print(f"问题：{question}")
    print(f"回答：{user_answer}")
    print()

    # TODO: 运行 Forum Agent
    # agent = ForumAgent(llm, storage_manager, db, embedding)
    # result = asyncio.run(agent.evaluate_answer(question, user_answer))
    # print(result)


if __name__ == "__main__":
    main()
