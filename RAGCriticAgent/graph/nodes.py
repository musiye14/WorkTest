"""
RAG Critic Agent 节点定义
"""

from .state import RAGCriticState
from ..prompt.prompt import RAG_CRITIC_SYSTEM_PROMPT, RAG_COMMENT_GENERATION_PROMPT, output_schema_rag_comment
from typing import Dict, Any, List
from storage.manager import StorageManager
from storage.database.postgresql import PostgreSQLDatabase
from rag.embedding import YEmbedding
import json


class RAGCriticNodes:
    """RAG Critic Agent 节点类，封装所有节点逻辑"""

    def __init__(
        self,
        llm,
        storage_manager: StorageManager,
        db: PostgreSQLDatabase,
        embedding: YEmbedding,
        top_k: int = 3,
        agent_name: str = "RAGCriticAgent"
    ):
        """
        初始化 RAG Critic 节点

        Args:
            llm: LLM 实例
            storage_manager: 存储管理器
            db: PostgreSQL 数据库实例
            embedding: 嵌入模型实例
            top_k: 检索相似案例的数量（默认 3）
            agent_name: Agent 名称，用于日志记录
        """
        self.llm = llm
        self.storage_manager = storage_manager
        self.db = db
        self.embedding = embedding
        self.top_k = top_k
        self.agent_name = agent_name

    async def search_similar_cases(self, state: RAGCriticState) -> Dict[str, Any]:
        """
        检索相似案例节点

        从 episodic_memory 检索相似的面试案例
        """
        question = state.get("question", "")
        interview_context = state.get("interview_context") or {}

        company = interview_context.get("company")
        difficulty = interview_context.get("difficulty")

        # 1. 将问题转换为 embedding
        query_embedding = self.embedding.embed_query(question)

        # 2. 在 Milvus 中检索
        milvus = self.storage_manager.get_milvus()

        # 构建过滤表达式
        filter_conditions = []

        if company:
            filter_conditions.append(f'company == "{company}"')

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
            return {"similar_cases": []}

        doc_ids = [result.document.id for result in search_results]

        # 4. 从 PostgreSQL 批量查询完整数据
        memories = await self.db.get_episodic_memory_by_ids(doc_ids)

        # 5. 按照相似度排序返回
        id_to_memory = {str(m['id']): m for m in memories}
        sorted_memories = [id_to_memory[doc_id] for doc_id in doc_ids if doc_id in id_to_memory]

        return {"similar_cases": sorted_memories}

    async def generate_comment(self, state: RAGCriticState) -> Dict[str, Any]:
        """
        生成评论节点

        基于检索到的相似案例，使用 LLM 生成评论
        """
        question = state.get("question", "")
        user_answer = state.get("user_answer", "")
        similar_cases = state.get("similar_cases", [])

        # 1. 格式化相似案例
        formatted_cases = self._format_similar_cases(similar_cases)

        # 2. 构建提示词
        prompt = RAG_COMMENT_GENERATION_PROMPT.format(
            question=question,
            user_answer=user_answer,
            similar_cases=formatted_cases
        )

        # 3. 调用 LLM（使用 invoke_with_schema）
        try:
            result = self.llm.invoke_with_schema(
                prompt=prompt,
                schema=output_schema_rag_comment,
                system_prompt=RAG_CRITIC_SYSTEM_PROMPT,
                node_name="generate_comment"
            )

            return {"rag_comment": result}

        except Exception as e:
            # 生成失败，返回错误信息
            return {
                "rag_comment": {
                    "error": "生成评论时发生错误",
                    "exception": str(e)
                }
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
