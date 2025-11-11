from typing_extensions import TypedDict
from typing import List,Dict,Annotated
from langgraph.graph.message import add_messages

class InterviewState(TypedDict):
    session_id:str
    user_id:str
    jd_info:str
    resume_info:str
    # 模式 real,training 训练的话是每次回答 直接获得面试官的点评
    mode:str 
    # 难度 "大厂"  "中厂"  "小厂"
    difficulty:str
    # 主问题列表
    question_plan:List[Dict]
    # 当前问题
    current_question:Dict
    # 历史对话
    messages:Annotated[list,add_messages]
    # 控制信号 追问"deep" or 结束 "end" or 下一个问题 "question" or 继续深度思考 "think"
    next_step:str

    # 追问轮次 比如设计成一个主问题，不能超过3个追问
    deep_index:int

    # 追问问题列表 追问的问题会重写这个列表
    deep_questions:List[Dict]

    # 思考的结果
    thinking_result:Dict
    # 思考的上下文 用于判断是不是可以结束反思
    thinking_process:List[Dict]

    # 反思结果
    reflection_result:List[Dict]

    # agent阶段
    interview_stage:str
