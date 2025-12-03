"""
RAG Critic Agent - 基于历史面经数据的评论家Agent
负责从 episodic_memory 检索相似案例并生成评论
"""

import json
import time
from typing import Dict, Any, Optional, List
from .tools import initialize_tools, get_rag_critic_tools
from .prompt import RAG_CRITIC_SYSTEM_PROMPT, RAG_COMMENT_GENERATION_PROMPT
from storage.manager import StorageManager
from storage.database.postgresql import PostgreSQLDatabase
from rag.embedding import YEmbedding
from langchain_core.messages import HumanMessage, SystemMessage


class RAGCriticAgent:
    """
    RAG Critic Agent - 基于历史面经数据的评论家

    职责：
    1. 从 episodic_memory 检索相似的面试案例
    2. 对比用户回答与标准答案
    3. 生成结构化评论（完整性、准确性、深度）
    4. 提供具体的改进建议
    """

    def __init__(
        self,
        llm,
        storage_manager: StorageManager,
        db: PostgreSQLDatabase,
        embedding: YEmbedding,
        top_k: int = 3
    ):
        """
        初始化 RAG Critic Agent

        Args:
            llm: LLM 实例（用于生成评论）
            storage_manager: 存储管理器
            db: PostgreSQL 数据库实例
            embedding: 嵌入模型实例
            top_k: 检索相似案例的数量（默认 3）
        """
        self.llm = llm
        self.storage_manager = storage_manager
        self.db = db
        self.embedding = embedding
        self.top_k = top_k

        # 初始化工具
        initialize_tools(storage_manager, db, embedding)
        self.tools = get_rag_critic_tools()

    async def generate_comment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成 RAG Critic 评论（作为 LangGraph 节点函数）

        Args:
            state: ForumState（包含 message、interview_context 等）

        Returns:
            更新后的 state（包含 rag_critic_comment）
        """
        print("\n" + "="*60)
        print("[RAG Critic Agent] 开始工作...")
        print("="*60)

        start_time = time.time()

        # 1. 从 state 中提取信息
        message = state.get("message", "")
        interview_context = state.get("interview_context") or {}

        # 2. 解析 message 提取问题和用户回答
        question, user_answer = self._parse_message(message)

        if not question or not user_answer:
            # 如果无法解析，返回空评论
            state["rag_critic_comment"] = {
                "error": "无法解析问题和用户回答",
                "message": message
            }
            return state

        # 3. 从 episodic_memory 检索相似案例
        print("\n[RAG Critic Agent] 正在检索相似案例...")
        search_start = time.time()

        user_id = state.get("user_id")  # 获取用户ID
        company = interview_context.get("company")
        difficulty = interview_context.get("difficulty")

        similar_cases = await self._search_similar_cases(
            question=question,
            user_id=user_id,
            company=company,
            difficulty=difficulty
        )

        search_duration = time.time() - search_start
        print(f"[RAG Critic Agent] 检索完成 | 找到 {len(similar_cases)} 个相似案例 | 耗时: {search_duration:.2f}s")

        # 打印检索到的案例详情
        if similar_cases:
            print("\n[RAG Critic Agent] 检索到的相似案例:")
            for i, case in enumerate(similar_cases, 1):
                print(f"  案例 {i}:")
                print(f"    - 问题: {case.get('abstract_question', 'N/A')[:50]}...")
                print(f"    - 难度: {case.get('difficulty', 'N/A')}")
                print(f"    - 质量评分: {case.get('quality_score', 'N/A')}/10")

        # 4. 使用 LLM 生成评论
        print("\n[RAG Critic Agent] 正在生成评论...")
        comment = await self._generate_comment_with_llm(
            question=question,
            user_answer=user_answer,
            similar_cases=similar_cases
        )

        # 5. 更新 state
        state["rag_critic_comment"] = comment

        total_duration = time.time() - start_time
        print(f"\n[RAG Critic Agent] 工作完成 | 总耗时: {total_duration:.2f}s")
        print("="*60 + "\n")

        return state

    def _parse_message(self, message: str) -> tuple[Optional[str], Optional[str]]:
        """
        从 message 中提取问题和用户回答

        message 格式示例：
        "面试官：请介绍一下Redis的持久化机制？\n用户：Redis有两种持久化方式..."

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
                # 提取最后一个问题
                question = line.split("：", 1)[1].strip()
            elif line.startswith("用户：") or line.startswith("候选人："):
                # 提取最后一个回答
                user_answer = line.split("：", 1)[1].strip()

        return question, user_answer

    async def _search_similar_cases(
        self,
        question: str,
        user_id: Optional[str] = None,
        company: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        从 episodic_memory 检索相似案例

        Args:
            question: 面试问题
            user_id: 用户UUID（只检索该用户的知识库）
            company: 公司名称（可选）
            difficulty: 难度（可选）

        Returns:
            相似案例列表
        """
        # 1. 将问题转换为 embedding
        query_embedding = self.embedding.embed_query(question)

        # 2. 在 Milvus 中检索
        milvus = self.storage_manager.get_milvus()

        # 构建过滤表达式
        filter_conditions = []

        # 按用户ID过滤（只检索该用户的知识库）
        if user_id:
            filter_conditions.append(f'user_id == "{user_id}"')

        # 注意：company 字段不在 metadata 中，因为面经知识点是通用的，不针对特定公司
        # 只使用 difficulty 和 quality_score 进行过滤

        if difficulty:
            filter_conditions.append(f'difficulty == "{difficulty}"')

        # 过滤质量评分（只返回高质量案例）
        filter_conditions.append('quality_score >= 7')

        filter_expr = ' and '.join(filter_conditions) if filter_conditions else 'quality_score >= 7'

        search_results = milvus.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_expr=filter_expr
        )

        # 3. 获取 doc_ids
        if not search_results:
            return []

        doc_ids = [result.document.id for result in search_results]

        # 4. 从 PostgreSQL 批量查询完整数据
        memories = await self.db.get_episodic_memory_by_ids(doc_ids)

        # 5. 按照相似度排序返回
        id_to_memory = {str(m['id']): m for m in memories}
        sorted_memories = [id_to_memory[doc_id] for doc_id in doc_ids if doc_id in id_to_memory]

        return sorted_memories

    async def _generate_comment_with_llm(
        self,
        question: str,
        user_answer: str,
        similar_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        使用 LLM 生成评论

        Args:
            question: 面试问题
            user_answer: 用户回答
            similar_cases: 相似案例列表

        Returns:
            结构化评论（JSON格式）
        """
        # 1. 格式化相似案例
        formatted_cases = self._format_similar_cases(similar_cases)

        # 2. 构建提示词
        prompt = RAG_COMMENT_GENERATION_PROMPT.format(
            question=question,
            user_answer=user_answer,
            similar_cases=formatted_cases
        )

        # 3. 调用 LLM
        messages = [
            SystemMessage(content=RAG_CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        try:
            llm_start = time.time()
            response = await self.llm.ainvoke(messages)
            llm_duration = time.time() - llm_start

            # 提取 token 使用量
            usage = response.usage if hasattr(response, 'usage') else {}
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)

            # 打印 token 统计
            print(f"[RAG Critic Agent] LLM调用完成 | "
                  f"输入token: {input_tokens} | "
                  f"输出token: {output_tokens} | "
                  f"总计: {total_tokens} | "
                  f"耗时: {llm_duration:.2f}s")

            # 4. 解析 JSON 响应
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 尝试提取 JSON（如果 LLM 返回了额外文本）
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                comment = json.loads(json_str)
            else:
                comment = json.loads(response_text)

            return comment

        except json.JSONDecodeError as e:
            # JSON 解析失败，返回错误信息
            return {
                "error": "LLM 返回的 JSON 格式不正确",
                "raw_response": response_text,
                "parse_error": str(e)
            }

        except Exception as e:
            # 其他错误
            return {
                "error": "生成评论时发生错误",
                "exception": str(e)
            }

    def _format_similar_cases(self, similar_cases: List[Dict[str, Any]]) -> str:
        """
        格式化相似案例为可读文本

        Args:
            similar_cases: 相似案例列表

        Returns:
            格式化后的文本
        """
        if not similar_cases:
            return "未找到相似案例"

        formatted = []

        for i, case in enumerate(similar_cases, 1):
            case_text = f"""
### 案例 {i}
- **问题**：{case.get('question', 'N/A')}
- **标准答案**：{case.get('answer', 'N/A')}
- **关键点**：{', '.join(case.get('key_points', []))}
- **公司**：{case.get('company', 'N/A')}
- **难度**：{case.get('difficulty', 'N/A')}
- **质量评分**：{case.get('quality_score', 'N/A')}/10
"""
            formatted.append(case_text.strip())

        return "\n\n".join(formatted)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行 RAG Critic Agent（便捷方法）

        Args:
            state: ForumState

        Returns:
            更新后的 state
        """
        return await self.generate_comment(state)
