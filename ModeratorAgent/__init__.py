"""
Moderator Agent - 主持人Agent
负责：
1. 初始化讨论
2. 决定讨论是否继续（多轮控制）
3. 汇总各方评论生成最终评价
"""
from .agent import ModeratorAgent

__all__ = ['ModeratorAgent']
