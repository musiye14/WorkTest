"""
Web Critic Agent - 网络评论家Agent
负责：
1. 使用 Tavily API 联网搜索
2. 获取最新技术资料和行业实践
3. 基于搜索结果生成评论
"""
from .agent import WebCriticAgent
from .tools import get_web_critic_tools, initialize_tools

__all__ = ['WebCriticAgent', 'get_web_critic_tools', 'initialize_tools']
