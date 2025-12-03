"""
Web Critic Agent - 基于网络搜索的评论家Agent
负责使用 Tavily API 获取最新技术资料并生成评论
"""

import json
import time
from typing import Dict, Any, Optional, List
from .tools import initialize_tools, get_web_critic_tools
from .prompt import WEB_CRITIC_SYSTEM_PROMPT, WEB_COMMENT_GENERATION_PROMPT
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, SystemMessage


class WebCriticAgent:
    """
    Web Critic Agent - 基于网络搜索的评论家

    职责：
    1. 使用 Tavily API 搜索最新技术资料
    2. 对比用户回答与当前行业实践
    3. 识别过时的技术观点
    4. 提供基于最新趋势的建议
    """

    def __init__(
        self,
        llm,
        tavily_api_key: str,
        max_search_results: int = 5
    ):
        """
        初始化 Web Critic Agent

        Args:
            llm: LLM 实例（用于生成评论）
            tavily_api_key: Tavily API Key
            max_search_results: 搜索结果数量（默认 5）
        """
        self.llm = llm
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.max_search_results = max_search_results

        # 初始化工具
        initialize_tools(tavily_api_key)
        self.tools = get_web_critic_tools()

    async def generate_comment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成 Web Critic 评论（作为 LangGraph 节点函数）

        Args:
            state: ForumState（包含 message、interview_context 等）

        Returns:
            更新后的 state（包含 web_critic_comment）
        """
        print("\n" + "="*60)
        print("[Web Critic Agent] 开始工作...")
        print("="*60)

        start_time = time.time()

        # 1. 从 state 中提取信息
        message = state.get("message", "")

        # 2. 解析 message 提取问题和用户回答
        question, user_answer = self._parse_message(message)

        if not question or not user_answer:
            # 如果无法解析，返回空评论
            state["web_critic_comment"] = {
                "error": "无法解析问题和用户回答",
                "message": message
            }
            return state

        # 3. 使用 Tavily API 搜索最新资料
        print("\n[Web Critic Agent] 正在搜索网络资料...")
        search_start = time.time()

        search_results = await self._search_web(question)

        search_duration = time.time() - search_start
        print(f"[Web Critic Agent] 搜索完成 | 找到 {len(search_results)} 个结果 | 耗时: {search_duration:.2f}s")

        # 打印搜索结果详情
        if search_results and 'error' not in search_results[0]:
            print("\n[Web Critic Agent] 搜索到的网络资料:")
            for i, result in enumerate(search_results, 1):
                print(f"  结果 {i}:")
                print(f"    - 标题: {result.get('title', 'N/A')[:50]}...")
                print(f"    - 来源: {result.get('url', 'N/A')[:60]}...")
                print(f"    - 相关性: {result.get('score', 0.0):.2f}")

        # 4. 使用 LLM 生成评论
        print("\n[Web Critic Agent] 正在生成评论...")
        comment = await self._generate_comment_with_llm(
            question=question,
            user_answer=user_answer,
            search_results=search_results
        )

        # 5. 更新 state
        state["web_critic_comment"] = comment

        total_duration = time.time() - start_time
        print(f"\n[Web Critic Agent] 工作完成 | 总耗时: {total_duration:.2f}s")
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

    async def _search_web(self, question: str) -> List[Dict[str, Any]]:
        """
        使用 Tavily API 搜索相关技术资料

        Args:
            question: 面试问题

        Returns:
            搜索结果列表
        """
        try:
            # 调用 Tavily API（使用 advanced 深度搜索）
            response = self.tavily_client.search(
                query=question,
                search_depth="advanced",
                max_results=self.max_search_results
            )

            # 提取搜索结果
            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'url': item.get('url', ''),
                    'score': item.get('score', 0.0),
                    'published_date': item.get('published_date', '')
                })

            return results

        except Exception as e:
            # 搜索失败，返回错误信息
            return [{
                'error': f"Tavily API 调用失败: {str(e)}"
            }]

    async def _generate_comment_with_llm(
        self,
        question: str,
        user_answer: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        使用 LLM 生成评论

        Args:
            question: 面试问题
            user_answer: 用户回答
            search_results: 网络搜索结果

        Returns:
            结构化评论（JSON格式）
        """
        # 1. 格式化搜索结果
        formatted_results = self._format_search_results(search_results)

        # 2. 构建提示词
        prompt = WEB_COMMENT_GENERATION_PROMPT.format(
            question=question,
            user_answer=user_answer,
            web_search_results=formatted_results
        )

        # 3. 调用 LLM
        messages = [
            SystemMessage(content=WEB_CRITIC_SYSTEM_PROMPT),
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
            print(f"[Web Critic Agent] LLM调用完成 | "
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

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果为可读文本

        Args:
            search_results: 搜索结果列表

        Returns:
            格式化后的文本
        """
        if not search_results:
            return "未找到相关搜索结果"

        # 检查是否有错误
        if len(search_results) == 1 and 'error' in search_results[0]:
            return f"搜索失败：{search_results[0]['error']}"

        formatted = []

        for i, result in enumerate(search_results, 1):
            result_text = f"""
### 搜索结果 {i}
- **标题**：{result.get('title', 'N/A')}
- **内容**：{result.get('content', 'N/A')}
- **来源**：{result.get('url', 'N/A')}
- **相关性评分**：{result.get('score', 0.0):.2f}
- **发布日期**：{result.get('published_date', 'N/A')}
"""
            formatted.append(result_text.strip())

        return "\n\n".join(formatted)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行 Web Critic Agent（便捷方法）

        Args:
            state: ForumState

        Returns:
            更新后的 state
        """
        return await self.generate_comment(state)
