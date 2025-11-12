from langgraph.graph import StateGraph
from .state import InterviewState
from .nodes import InterviewNodes
from ..llms.base import BaseLLM


def buildGraph(llm: BaseLLM, think_max_num: int = 3, deep_question_max_num: int = 3):
    """
    构建面试 Agent 图

    Args:
        llm: LLM 实例
        think_max_num: 思考最大轮次，默认 3
        deep_question_max_num: 追问最大次数，默认 3

    Returns:
        编译后的图
    """
    nodes = InterviewNodes(llm, think_max_num=think_max_num, deep_question_max_num=deep_question_max_num)
    builder = StateGraph(InterviewState)

    # 添加节点
    builder.add_node("messageInput", nodes.message_input)
    builder.add_node("questionBuild", nodes.question_build)
    builder.add_node("questionBuildThink", nodes.think)
    builder.add_node("questionJudge", nodes.judge)
    builder.add_node("questionOutput", nodes.question_output)
    builder.add_node("nextStep", nodes.next_step)
    builder.add_node("adjustQuestion", nodes.adjust_question)
    builder.add_node("adjustQuestionThink", nodes.think)
    builder.add_node("adjustQuestionJudge", nodes.judge)
    builder.add_node("deepQuestion", nodes.deep_question)
    builder.add_node("deepQuestionThink", nodes.think)
    builder.add_node("deepQuestionJudge", nodes.judge)
    builder.add_node("deepQuestionOutput", nodes.deep_question_output)
    builder.add_node("userInput", nodes.user_input)
    builder.add_node("end", nodes.end)

    # ========== 主流程边 ==========
    builder.add_edge("messageInput", "questionBuild")
    builder.add_edge("questionBuild", "questionBuildThink")
    builder.add_edge("questionBuildThink", "questionJudge")
    builder.add_conditional_edges("questionJudge", nodes.judge_res)
    builder.add_edge("questionOutput", "userInput")
    builder.add_edge("userInput", "nextStep")
    builder.add_conditional_edges("nextStep", nodes.next_step_decision)

    # ========== 调整问题分支 ==========
    builder.add_edge("adjustQuestion", "adjustQuestionThink")
    builder.add_edge("adjustQuestionThink", "adjustQuestionJudge")
    builder.add_conditional_edges("adjustQuestionJudge", nodes.judge_res)

    # ========== 追问分支 ==========
    builder.add_edge("deepQuestion", "deepQuestionThink")
    builder.add_edge("deepQuestionThink", "deepQuestionJudge")
    builder.add_conditional_edges("deepQuestionJudge", nodes.judge_res)
    builder.add_edge("deepQuestionOutput", "userInput")

    # ========== 入口点 ==========
    builder.set_entry_point("messageInput")

    return builder.compile()
