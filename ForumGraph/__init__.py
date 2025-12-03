"""
Forum Graph - 论坛讨论图协调层
负责：
1. 定义共享的 ForumState
2. 构建 LangGraph 协调三个Agent
3. 管理多轮讨论流程
"""
from .graph.state import ForumState
from .graph.graph import buildGraph
from .agent import ForumAgent

__all__ = ['ForumState', 'buildGraph', 'ForumAgent']
