"""
面试 Agent 主类
负责初始化、用户交互和面试流程控制
"""

import uuid
import sys
import os
from typing import Dict, Any, Optional
from .llms.base import BaseLLM
from .graph.graph import buildGraph
from .graph.state import InterviewState

from config import get_config


class InterviewAgent:
    """面试 Agent 主类"""

    def __init__(self, llm: BaseLLM, user_id: Optional[str] = None, think_max_num: Optional[int] = None, deep_question_max_num: Optional[int] = None):
        """
        初始化面试 Agent

        Args:
            llm: LLM 实例
            user_id: 用户 ID（可选，默认生成随机 ID）
            think_max_num: 思考最大轮次（可选，默认从配置读取）
            deep_question_max_num: 追问最大次数（可选，默认从配置读取）
        """
        self.llm = llm
        self.user_id = user_id or self._generate_user_id()

        # 从配置读取参数，如果未提供
        config = get_config()
        if think_max_num is None:
            think_max_num = config.get('THINK_MAX_NUM', 3)
        if deep_question_max_num is None:
            deep_question_max_num = config.get('DEEP_QUESTION_MAX_NUM', 3)

        self.think_max_num = think_max_num
        self.deep_question_max_num = deep_question_max_num
        self.graph = buildGraph(llm, think_max_num=think_max_num, deep_question_max_num=deep_question_max_num)
        self.current_state: Optional[InterviewState] = None

    @staticmethod
    def _generate_user_id() -> str:
        """生成随机用户 ID"""
        return f"user_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _generate_session_id() -> str:
        """生成随机会话 ID"""
        return f"session_{uuid.uuid4().hex[:12]}"

    def collect_interview_info(self) -> Dict[str, Any]:
        """
        交互式收集面试信息

        Returns:
            包含 JD、简历、模式、难度的字典
        """
        print("=" * 50)
        print("欢迎使用面试 Agent")
        print("=" * 50)
        print()

        # 收集 JD 信息
        print("请输入 JD 信息（职位描述）：")
        print("（可以粘贴完整的 JD，输入完成后按回车）")
        jd_info = input("> ").strip()
        print()

        # 收集简历信息
        print("请输入简历信息：")
        print("（可以粘贴完整的简历，输入完成后按回车）")
        resume_info = input("> ").strip()
        print()

        # 选择模式
        print("请选择面试模式：")
        print("1. real - 真实面试模式")
        print("2. training - 训练模式（会提供即时反馈）")
        mode_choice = input("请输入选项（1 或 2）: ").strip()
        mode = "real" if mode_choice == "1" else "training"
        print(f"已选择：{mode} 模式")
        print()

        # 选择难度
        print("请选择公司难度：")
        print("1. 大厂（15-20 个问题，50% 困难）")
        print("2. 中厂（10-15 个问题，30% 困难）")
        print("3. 小厂（8-12 个问题，20% 困难）")
        difficulty_choice = input("请输入选项（1/2/3）: ").strip()
        difficulty_map = {"1": "大厂", "2": "中厂", "3": "小厂"}
        difficulty = difficulty_map.get(difficulty_choice, "中厂")
        print(f"已选择：{difficulty}")
        print()

        return {
            "jd_info": jd_info,
            "resume_info": resume_info,
            "mode": mode,
            "difficulty": difficulty
        }

    def create_initial_state(self, interview_info: Dict[str, Any]) -> InterviewState:
        """
        创建初始状态

        Args:
            interview_info: 面试信息（JD、简历、模式、难度）

        Returns:
            初始化的 InterviewState
        """
        return {
            "session_id": self._generate_session_id(),
            "user_id": self.user_id,
            "jd_info": interview_info["jd_info"],
            "resume_info": interview_info["resume_info"],
            "mode": interview_info["mode"],
            "difficulty": interview_info["difficulty"],
            "question_plan": [],
            "current_question": {},
            "messages": [],
            "next_step": "",
            "deep_index": 0,
            "deep_questions": [],
            "thinking_result": {},
            "thinking_process": [],
            "interview_stage": ""
        }

    def run_interview(self, interview_info: Optional[Dict[str, Any]] = None):
        """
        运行完整的面试流程（流式交互）

        Args:
            interview_info: 面试信息（可选，如果不提供则交互式收集）
        """
        # 收集面试信息
        if interview_info is None:
            interview_info = self.collect_interview_info()

        # 创建初始状态
        initial_state = self.create_initial_state(interview_info)

        print("=" * 50)
        print("面试准备完成，开始生成问题...")
        print("=" * 50)
        print()

        # 流式执行图
        try:
            # 配置递归限制，避免无限循环
            config = {"recursion_limit": 500}

            for event in self.graph.stream(initial_state, config=config):
                # event 是 {node_name: output} 的字典
                for node_name, output in event.items():
                    print(f"[调试] 节点 {node_name} 执行完成")

                    # 如果到达结束节点
                    if output and output.get("next_step") == "end":
                        print("\n" + "=" * 50)
                        print("面试结束，感谢参与！")
                        print("=" * 50)
                        return

        except Exception as e:
            print(f"面试过程中发生错误：{e}")
            import traceback
            traceback.print_exc()
            raise

    def get_interview_history(self) -> list:
        """
        获取面试历史对话

        Returns:
            消息列表
        """
        if self.current_state is None:
            return []
        return self.current_state.get("messages", [])


def main():
    """命令行交互式面试入口"""
    from .llms.openai_llm import OpenAILLM
    import os

    # 从环境变量获取 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        print("请设置环境变量：export OPENAI_API_KEY=your-api-key")
        return

    # 初始化 LLM
    llm = OpenAILLM(
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        base_url=base_url
    )

    # 创建 Agent
    agent = InterviewAgent(llm)

    # 启动面试
    try:
        agent.run_interview()
    except KeyboardInterrupt:
        print("\n\n面试被中断")
    except Exception as e:
        print(f"\n发生错误：{e}")


if __name__ == "__main__":
    main()
