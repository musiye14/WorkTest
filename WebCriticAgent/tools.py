"""
Web Critic Agent 工具集
负责使用 Tavily API 进行网络搜索
"""

from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from tavily import TavilyClient
from config import get_config


# 全局 Tavily 客户端实例（需要在使用前初始化）
_tavily_client: Optional[TavilyClient] = None


def initialize_tools(tavily_api_key: Optional[str] = None):
    """
    初始化工具（在使用前必须调用）

    Args:
        tavily_api_key: Tavily API Key（如果不提供，从配置读取）
    """
    global _tavily_client

    if tavily_api_key is None:
        config = get_config()
        tavily_api_key = config.get('TAVILY_API_KEY')

    if not tavily_api_key:
        raise ValueError("未找到 TAVILY_API_KEY，请在 env 文件中配置")

    _tavily_client = TavilyClient(api_key=tavily_api_key)


@tool
def search_web_for_technical_info(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    使用 Tavily API 搜索最新技术资料

    使用场景：
    - 查找技术的最新发展和行业实践
    - 验证用户回答的技术准确性
    - 获取官方文档和权威资料

    参数：
        query: 搜索查询（技术点或问题）
        search_depth: 搜索深度（"basic" 或 "advanced"，默认 "advanced"）
        max_results: 最大返回结果数（默认 5）

    返回：
        搜索结果列表，包含标题、内容、URL、相关性评分
    """
    if not _tavily_client:
        raise RuntimeError("工具未初始化，请先调用 initialize_tools()")

    try:
        # 调用 Tavily API
        response = _tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results
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
        # 捕获 API 调用错误
        return [{
            'error': f"Tavily API 调用失败: {str(e)}"
        }]


@tool
def search_web_for_best_practices(
    topic: str,
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """
    搜索特定技术的最佳实践

    使用场景：
    - 了解行业标准做法
    - 查找权威的技术建议
    - 对比用户回答与业界实践

    参数：
        topic: 技术主题（如 "Redis 缓存最佳实践"）
        max_results: 最大返回结果数（默认 3）

    返回：
        搜索结果列表
    """
    # 构建更精确的查询
    query = f"{topic} best practices OR 最佳实践 OR 实战经验"

    return search_web_for_technical_info(
        query=query,
        search_depth="advanced",
        max_results=max_results
    )


# 导出所有工具
WEB_CRITIC_TOOLS = [
    search_web_for_technical_info,
    search_web_for_best_practices
]


def get_web_critic_tools() -> List:
    """获取 Web Critic Agent 的所有工具"""
    return WEB_CRITIC_TOOLS
