from .state import InterviewState
from ..prompt.prompt import *
from ..llms.base import BaseLLM
from typing import Dict, Any
from langchain_core.messages import HumanMessage,AIMessage


class InterviewNodes:
    """面试节点类，封装所有节点逻辑"""

    def __init__(self, llm: BaseLLM, think_max_num: int = 3, deep_question_max_num: int = 3):
        """
        初始化面试节点

        Args:
            llm: LLM 实例
            think_max_num: 思考最大轮次，默认 3
            deep_question_max_num: 追问最大次数，默认 3
        """
        self.llm = llm
        self.think_max_num = think_max_num
        self.deep_question_max_num = deep_question_max_num

    def message_input(self, state: InterviewState) -> Dict[str, Any]:
        """输入处理节点"""
        return {
            "interview_stage": "questionBuild",
            "thinking_process": [],
            "deep_index": 0
        }

    def question_build(self, state: InterviewState) -> Dict[str, Any]:
        """问题生成节点"""
        # 获取上一轮的反馈（如果有）
        reflection_feedback = state.get("reflection_result", {})
        improvement_suggestions = reflection_feedback.get("improvement_suggestions", [])

        # 如果有反馈，使用带反馈的提示词；否则使用普通提示词
        if improvement_suggestions:
            feedback_text = "\n".join([f"- {s}" for s in improvement_suggestions])
            prompt = QUESTION_PLAN_GENERATION_WITH_FEEDBACK_PROMPT.format(
                resume=state["resume_info"],
                jd=state["jd_info"],
                mode=state["mode"],
                difficulty=state["difficulty"],
                previous_attempt=state.get("question_plan", []),
                feedback=feedback_text
            )
        else:
            prompt = QUESTION_PLAN_GENERATION_PROMPT.format(
                resume=state["resume_info"],
                jd=state["jd_info"],
                mode=state["mode"],
                difficulty=state["difficulty"]
            )

        result = self.llm.invoke_with_schema(prompt, output_schema_question_plan_generation)
        return {"question_plan": result["question_plan"]}

    def think(self, state: InterviewState) -> Dict[str, Any]:
        """深度思考节点，根据 interview_stage 选择对应的深度思考提示词"""
        round_num = len(state["thinking_process"]) + 1
        stage = state["interview_stage"]

        if stage == "questionBuild":
            prompt = DEEP_THINKING_PROMPT_QUESTION_BUILD.format(
                resume=state["resume_info"],
                jd=state["jd_info"],
                mode=state["mode"],
                difficulty=state["difficulty"],
                round=round_num
            )
        elif stage == "adjustQuestion":
            prompt = DEEP_THINKING_PROMPT_ADJUST_QUESTION.format(
                questions_plan=state["question_plan"],
                completed_questions=state.get("completed_questions", []),
                performance_analysis=state.get("performance_analysis", ""),
                weak_points=state.get("weak_points", []),
                strong_points=state.get("strong_points", []),
                round=round_num
            )
        elif stage == "deepQuestion":
            # 获取用户回答内容
            user_answer = ""
            if state["messages"]:
                last_msg = state["messages"][-1]
                if isinstance(last_msg, HumanMessage):
                    user_answer = last_msg.content

            prompt = DEEP_THINKING_PROMPT_DEEP_QUESTION.format(
                current_question=state["current_question"],
                user_answer=user_answer,
                answer_analysis=state.get("answer_analysis", ""),
                follow_up_count=state["deep_index"],
                difficulty=state["difficulty"],
                round=round_num
            )
        else:
            raise ValueError(f"未知的 interview_stage: {stage}")

        result = self.llm.invoke_with_schema(prompt, output_schema_deep_thinking)
        return {
            "thinking_result": result,
            "thinking_process": state["thinking_process"] + [result]
        }

    def judge(self, state: InterviewState) -> Dict[str, Any]:
        """反思判断节点，根据 interview_stage 选择对应的反思提示词"""
        thinking_process = state["thinking_process"]
        round_num = len(thinking_process)
        stage = state["interview_stage"]

        if stage == "questionBuild":
            prompt = REFLECTION_PROMPT_QUESTION_BUILD.format(
                draft_question_plan=state["question_plan"],
                thinking_process=state["thinking_process"],
                resume=state["resume_info"],
                jd=state["jd_info"],
                difficulty=state["difficulty"],
                round=round_num
            )
        elif stage == "adjustQuestion":
            prompt = REFLECTION_PROMPT_ADJUST_QUESTION.format(
                adjusted_question_plan=state["question_plan"],
                original_question_plan=state.get("original_question_plan", []),
                thinking_process=state["thinking_process"],
                performance_analysis=state.get("performance_analysis", ""),
                round=round_num
            )
        elif stage == "deepQuestion":
            # 获取用户回答内容
            user_answer = ""
            if state["messages"]:
                last_msg = state["messages"][-1]
                if isinstance(last_msg, HumanMessage):
                    user_answer = last_msg.content

            prompt = REFLECTION_PROMPT_DEEP_QUESTION.format(
                draft_deep_question=state.get("current_question", {}),
                thinking_process=state["thinking_process"],
                current_question=state["current_question"],
                user_answer=user_answer,
                follow_up_count=state["deep_index"],
                round=round_num
            )
        else:
            raise ValueError(f"未知的 interview_stage: {stage}")

        result = self.llm.invoke_with_schema(prompt, output_schema_reflection)

        # 检查是否超过最大轮次限制
        if round_num >= self.think_max_num:
            # 超过限制,强制通过(即使质量不够好)
            print(f"[警告] 思考轮次已达上限 ({round_num}/{self.think_max_num}),强制通过")
            result["should_regenerate"] = False
            # 添加说明到改进建议中
            if "improvement_suggestions" not in result:
                result["improvement_suggestions"] = []
            result["improvement_suggestions"].append(
                f"已达到最大思考轮次 ({self.think_max_num}),强制通过当前结果"
            )
            thinking_process = []


        return {"reflection_result": result, "thinking_process":thinking_process}

    def question_output(self, state: InterviewState) -> Dict[str, Any]:
        """提问输出节点 - 从问题列表中取出当前问题并输出"""
        question_plan = state.get("question_plan", [])
        # 统计已提问的数量（AIMessage 的数量）
        current_question_index = len([msg for msg in state.get("messages", []) if isinstance(msg, AIMessage)])

        # 如果还有问题未提问
        if current_question_index < len(question_plan):
            current_question = question_plan[current_question_index]

            print(f"面试官：{current_question.get('question', '')}")

            return {
                "current_question": current_question,
                "messages": [AIMessage(content=current_question["question"])]
            }
        else:
            # 所有问题已问完
            return {"next_step": "end"}

    def deep_question_output(self, state: InterviewState) -> Dict[str, Any]:
        """追问输出节点 - 输出追问问题给用户"""
        current_question = state.get("current_question", {})
        follow_up_question = current_question.get("question", "")
        
        print(f"面试官：{follow_up_question}")
        
        return {
            "messages": [AIMessage(content=follow_up_question)]
        }

    def user_input(self, state: InterviewState) -> Dict[str, Any]:
        """等待用户输入节点"""
        user_answer = input("你的回答：").strip()
        return {
            "messages": [HumanMessage(content=user_answer)]
        }

    def next_step(self, state: InterviewState) -> Dict[str, Any]:
        """决策下一步节点 - 分析用户回答并决定下一步动作"""
        # 获取最新的用户回答
        messages = state.get("messages", [])
        if not messages:
            return {"next_step": "end"}

        last_message = messages[-1]

        # 检查是否是用户消息
        if not isinstance(last_message, HumanMessage):
            return {"next_step": "end"}

        user_answer = last_message.content
        current_question = state.get("current_question", {})
        deep_index = state.get("deep_index", 0)

        # 调用 LLM 判断是否需要追问
        prompt = FOLLOW_UP_DECISION_PROMPT.format(
            current_question=current_question.get("question", ""),
            user_answer=user_answer,
            answer_analysis="",  # 可以添加回答分析逻辑
            follow_up_count=deep_index
        )

        decision = self.llm.invoke_with_schema(prompt, output_schema_follow_up_decision)

        # 根据决策结果设置下一步
        if decision.get("should_follow_up", False)==True and deep_index < self.deep_question_max_num:
            # 需要追问
            return {
                "next_step": "deepQuestion",
                "interview_stage": "deepQuestion",
            }
        else:
            # 不需要追问，进入下一个问题
            # 重置追问计数
            return {
                "next_step": "question",
                "deep_index": 0
            }

    def adjust_question(self, state: InterviewState) -> Dict[str, Any]:
        """调整问题节点"""
        # 获取上一轮的反馈（如果有）
        reflection_feedback = state.get("reflection_result", {})
        improvement_suggestions = reflection_feedback.get("improvement_suggestions", [])

        # 如果有反馈，使用带反馈的提示词；否则使用普通提示词
        if improvement_suggestions:
            feedback_text = "\n".join([f"- {s}" for s in improvement_suggestions])
            prompt = QUESTION_ADJUSTMENT_WITH_FEEDBACK_PROMPT.format(
                questions_plan=state["question_plan"],
                completed_questions=state.get("completed_questions", []),
                performance_analysis=state.get("performance_analysis", ""),
                weak_points=state.get("weak_points", []),
                strong_points=state.get("strong_points", []),
                difficulty=state["difficulty"],
                previous_attempt=state.get("question_plan", []),
                feedback=feedback_text
            )
        else:
            prompt = QUESTION_ADJUSTMENT_PROMPT.format(
                questions_plan=state["question_plan"],
                completed_questions=state.get("completed_questions", []),
                performance_analysis=state.get("performance_analysis", ""),
                weak_points=state.get("weak_points", []),
                strong_points=state.get("strong_points", []),
                difficulty=state["difficulty"]
            )

        result = self.llm.invoke_with_schema(prompt, output_schema_question_adjustment)
        return {
            "question_plan": result["adjusted_question_plan"],
            "original_question_plan": state["question_plan"]
        }

    def deep_question(self, state: InterviewState) -> Dict[str, Any]:
        """追问生成节点"""
        # 获取用户回答内容
        user_answer = ""
        if state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                user_answer = last_msg.content

        # 获取上一轮的反馈（如果有）
        reflection_feedback = state.get("reflection_result", {})
        improvement_suggestions = reflection_feedback.get("improvement_suggestions", [])

        # 如果有反馈，使用带反馈的提示词；否则使用普通提示词
        if improvement_suggestions:
            feedback_text = "\n".join([f"- {s}" for s in improvement_suggestions])
            prompt = DEEP_QUESTION_GENERATION_WITH_FEEDBACK_PROMPT.format(
                current_question=state["current_question"],
                user_answer=user_answer,
                answer_analysis=state.get("answer_analysis", ""),
                follow_up_count=state["deep_index"],
                difficulty=state["difficulty"],
                previous_attempt=state.get("current_question", {}),
                feedback=feedback_text
            )
        else:
            prompt = DEEP_QUESTION_GENERATION_PROMPT.format(
                current_question=state["current_question"],
                user_answer=user_answer,
                answer_analysis=state.get("answer_analysis", ""),
                follow_up_count=state["deep_index"],
                difficulty=state["difficulty"]
            )

        result = self.llm.invoke_with_schema(prompt, output_schema_deep_question_generation)
        return {
            "current_question": result,
            "deep_index": state["deep_index"] + 1
        }

    def judge_res(self, state: InterviewState) -> str:
        """反思判断结果路由"""
        reflection = state.get("reflection_result", {})
        stage = state["interview_stage"]

        if reflection.get("should_regenerate", False) == True:
            # 需要重新生成
            if stage == "questionBuild":
                return "questionBuild"
            elif stage == "adjustQuestion":
                return "adjustQuestion"
            elif stage == "deepQuestion":
                return "deepQuestion"
        else:
            # 通过检查，根据阶段输出对应的问题
            if stage == "deepQuestion":
                return "deepQuestionOutput"
            else:
                return "questionOutput"

    def next_step_decision(self, state: InterviewState) -> str:
        """下一步决策路由"""
        next_step = state.get("next_step", "")

        if next_step == "adjustQuestion":
            return "adjustQuestion"
        elif next_step == "deepQuestion":
            return "deepQuestion"
        elif next_step == "end":
            return "end"
        else:
            return "questionOutput"
