"""
假设性问题生成器 - 提升检索效果

原理：
1. 对文档使用 LLM 生成可能的问题
2. 将问题向量化存储
3. 用户查询时，问题匹配问题（而非问题匹配文档）
4. 提高检索准确率
"""
from typing import List, Dict, Optional


class HypotheticalQuestionGenerator:
    """
    假设性问题生成器

    功能：
    1. 为文档生成假设性问题
    2. 存储问题-文档映射关系
    """

    def __init__(self, llm=None):
        """
        初始化生成器

        参数:
            llm: 语言模型（可选）
        """
        self.llm = llm
        # TODO: 初始化 LLM

    def generate_questions(
        self,
        document: str,
        num_questions: int = 3
    ) -> List[str]:
        """
        为文档生成假设性问题

        参数:
            document: 文档内容
            num_questions: 生成问题数量

        返回:
            问题列表
        """
        # TODO: 使用 LLM 生成问题
        # prompt = f"""
        # 请为以下文档生成 {num_questions} 个可能的问题，这些问题应该是用户在搜索时可能会问的。
        #
        # 文档内容：
        # {document}
        #
        # 要求：
        # 1. 问题要具体、明确
        # 2. 问题要覆盖文档的核心内容
        # 3. 问题要符合用户的搜索习惯
        #
        # 请以 JSON 格式返回：
        # ["问题1", "问题2", "问题3"]
        # """

        # 临时实现：返回空列表
        return []

    def create_question_document_mapping(
        self,
        doc_id: str,
        doc_content: str,
        questions: List[str]
    ) -> List[Dict]:
        """
        创建问题-文档映射

        参数:
            doc_id: 文档 ID
            doc_content: 文档内容
            questions: 问题列表

        返回:
            映射列表（用于存储到向量数据库）
        """
        mappings = []
        for question in questions:
            mappings.append({
                "id": f"{doc_id}_q_{len(mappings)}",
                "content": question,  # 存储问题文本
                "metadata": {
                    "type": "hypothetical_question",
                    "original_doc_id": doc_id,
                    "original_doc_content": doc_content[:500]  # 存储部分原文
                }
            })
        return mappings
