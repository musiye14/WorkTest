"""
Moderator Agent - 主持人Agent
负责协调两位 Critic，决策流程，生成最终评价
"""

import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from .prompt import (
    MODERATOR_SYSTEM_PROMPT,
    MODERATOR_DECISION_PROMPT,
    MODERATOR_FINAL_EVALUATION_PROMPT
)
from langchain_core.messages import HumanMessage, SystemMessage


class ModeratorAgent:
    """
    Moderator Agent - 主持人

    职责：
    1. 协调 RAG Critic 和 Web Critic 的讨论
    2. 决定是否继续下一轮讨论
    3. 生成综合性的最终评价
    4. 管理讨论流程和轮次控制
    """

    def __init__(self, llm):
        """
        初始化 Moderator Agent

        Args:
            llm: LLM 实例（用于决策和生成最终评价）
        """
        self.llm = llm

    async def decide_next_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        决定下一步动作（作为 LangGraph 节点函数）

        Args:
            state: ForumState

        Returns:
            更新后的 state（包含 should_continue、next_step）
        """
        # 1. 提取当前状态信息
        current_round = state.get("current_round", 1)
        max_rounds = state.get("max_rounds", 3)
        current_speaker = state.get("current_speaker", "")
        rag_comment = state.get("rag_critic_comment")
        web_comment = state.get("web_critic_comment")

        # 2. 如果是第一轮且还没有 Critic 评论，启动 RAG Critic
        if current_round == 1 and not rag_comment:
            state["next_step"] = "rag_critic"
            state["current_speaker"] = "rag_critic"
            state["should_continue"] = True
            return state

        # 3. 如果 RAG Critic 刚评论完，启动 Web Critic
        if rag_comment and not web_comment:
            state["next_step"] = "web_critic"
            state["current_speaker"] = "web_critic"
            state["should_continue"] = True
            return state

        # 4. 如果两位 Critic 都评论完了，进行决策
        if rag_comment and web_comment:
            decision = await self._make_decision_with_llm(
                current_round=current_round,
                max_rounds=max_rounds,
                current_speaker=current_speaker,
                rag_comment=rag_comment,
                web_comment=web_comment
            )

            # 5. 更新状态
            state["should_continue"] = decision.get("should_continue", False)
            state["next_step"] = decision.get("next_step", "end")
            state["current_speaker"] = decision.get("current_speaker", "moderator")

            # 6. 如果决定继续，轮次+1，清空评论
            if state["should_continue"] and state["next_step"] == "rag_critic":
                # 保存当前轮次的讨论历史
                discussion_history = state.get("discussion_history", [])
                discussion_history.append({
                    "round": current_round,
                    "rag_comment": rag_comment,
                    "web_comment": web_comment,
                    "timestamp": datetime.now().isoformat()
                })
                state["discussion_history"] = discussion_history

                # 轮次+1
                state["current_round"] = current_round + 1

                # 清空当前评论，准备下一轮
                state["rag_critic_comment"] = None
                state["web_critic_comment"] = None

        return state

    async def generate_final_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成最终评价（作为 LangGraph 节点函数）

        Args:
            state: ForumState

        Returns:
            更新后的 state（包含 final_evaluation）
        """
        print("\n" + "="*60)
        print("[Moderator Agent] 开始生成最终评价...")
        print("="*60)

        start_time = time.time()

        # 1. 提取信息
        message = state.get("message", "")
        rag_comment = state.get("rag_critic_comment")
        web_comment = state.get("web_critic_comment")
        discussion_history = state.get("discussion_history", [])

        # 2. 解析问题和用户回答
        question, user_answer = self._parse_message(message)

        # 3. 使用 LLM 生成最终评价
        print("\n[Moderator Agent] 正在综合两位 Critic 的评论...")
        final_evaluation = await self._generate_final_evaluation_with_llm(
            question=question,
            user_answer=user_answer,
            rag_comment=rag_comment,
            web_comment=web_comment,
            discussion_history=discussion_history
        )

        # 4. 更新状态
        state["final_evaluation"] = final_evaluation
        state["next_step"] = "save"

        total_duration = time.time() - start_time
        print(f"\n[Moderator Agent] 最终评价生成完成 | 总耗时: {total_duration:.2f}s")
        print("="*60 + "\n")

        return state

    def _parse_message(self, message: str) -> tuple[Optional[str], Optional[str]]:
        """
        从 message 中提取问题和用户回答

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
                question = line.split("：", 1)[1].strip()
            elif line.startswith("用户：") or line.startswith("候选人："):
                user_answer = line.split("：", 1)[1].strip()

        return question, user_answer

    async def _make_decision_with_llm(
        self,
        current_round: int,
        max_rounds: int,
        current_speaker: str,
        rag_comment: Dict[str, Any],
        web_comment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用 LLM 进行决策

        Args:
            current_round: 当前轮次
            max_rounds: 最大轮次
            current_speaker: 当前发言者
            rag_comment: RAG Critic 的评论
            web_comment: Web Critic 的评论

        Returns:
            决策结果（JSON格式）
        """
        # 1. 格式化评论
        formatted_rag = json.dumps(rag_comment, ensure_ascii=False, indent=2)
        formatted_web = json.dumps(web_comment, ensure_ascii=False, indent=2)

        # 2. 构建提示词
        prompt = MODERATOR_DECISION_PROMPT.format(
            current_round=current_round,
            max_rounds=max_rounds,
            current_speaker=current_speaker,
            rag_critic_comment=formatted_rag,
            web_critic_comment=formatted_web
        )

        # 3. 调用 LLM
        messages = [
            SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 4. 解析 JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)
            else:
                decision = json.loads(response_text)

            return decision

        except Exception as e:
            # 决策失败，默认结束讨论
            return {
                "should_continue": False,
                "next_step": "moderator_summarize",
                "reason": f"决策失败：{str(e)}",
                "current_speaker": "moderator"
            }

    async def _generate_final_evaluation_with_llm(
        self,
        question: Optional[str],
        user_answer: Optional[str],
        rag_comment: Dict[str, Any],
        web_comment: Dict[str, Any],
        discussion_history: list
    ) -> Dict[str, Any]:
        """
        使用 LLM 生成最终评价

        Args:
            question: 面试问题
            user_answer: 用户回答
            rag_comment: RAG Critic 的评论
            web_comment: Web Critic 的评论
            discussion_history: 讨论历史

        Returns:
            最终评价（JSON格式）
        """
        # 1. 格式化评论和历史
        formatted_rag = json.dumps(rag_comment, ensure_ascii=False, indent=2)
        formatted_web = json.dumps(web_comment, ensure_ascii=False, indent=2)
        formatted_history = json.dumps(discussion_history, ensure_ascii=False, indent=2)

        # 2. 构建提示词
        prompt = MODERATOR_FINAL_EVALUATION_PROMPT.format(
            question=question or "未提取到问题",
            user_answer=user_answer or "未提取到回答",
            rag_critic_comment=formatted_rag,
            web_critic_comment=formatted_web,
            discussion_history=formatted_history
        )

        # 3. 调用 LLM
        messages = [
            SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
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
            print(f"[Moderator Agent] LLM调用完成 | "
                  f"输入token: {input_tokens} | "
                  f"输出token: {output_tokens} | "
                  f"总计: {total_tokens} | "
                  f"耗时: {llm_duration:.2f}s")

            response_text = response.content if hasattr(response, 'content') else str(response)

            # 4. 解析 JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
            else:
                evaluation = json.loads(response_text)

            return evaluation

        except json.JSONDecodeError as e:
            return {
                "error": "LLM 返回的 JSON 格式不正确",
                "raw_response": response_text,
                "parse_error": str(e)
            }

        except Exception as e:
            return {
                "error": "生成最终评价时发生错误",
                "exception": str(e)
            }

    async def generate_overall_evaluation(
        self,
        qa_evaluations: List[Dict[str, Any]],
        interview_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于多个 QA 的评价，生成总体评价

        Args:
            qa_evaluations: 每条 QA 的评价列表
            interview_context: 面试上下文（公司、岗位、难度等）

        Returns:
            总体评价（JSON格式）
        """
        from .prompt import MODERATOR_OVERALL_EVALUATION_PROMPT

        # 1. 提取所有评分
        scores = []
        for qa in qa_evaluations:
            evaluation = qa.get('evaluation', {})
            if isinstance(evaluation, dict):
                overall_score = evaluation.get('overall_score', 0)
                scores.append(overall_score)

        # 2. 计算平均分
        average_score = sum(scores) / len(scores) if scores else 0

        # 3. 格式化 qa_evaluations 为 JSON 字符串
        formatted_qa_evaluations = json.dumps(qa_evaluations, ensure_ascii=False, indent=2)
        formatted_interview_context = json.dumps(interview_context, ensure_ascii=False, indent=2)

        # 4. 构建提示词
        prompt = MODERATOR_OVERALL_EVALUATION_PROMPT.format(
            total_questions=len(qa_evaluations),
            average_score=average_score,
            interview_context=formatted_interview_context,
            qa_evaluations=formatted_qa_evaluations
        )

        # 5. 调用 LLM 生成总体评价
        messages = [
            SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
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
            print(f"[Moderator Agent] 总体评价LLM调用完成 | "
                  f"输入token: {input_tokens} | "
                  f"输出token: {output_tokens} | "
                  f"总计: {total_tokens} | "
                  f"耗时: {llm_duration:.2f}s")

            response_text = response.content if hasattr(response, 'content') else str(response)

            # 6. 解析 JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                overall_evaluation = json.loads(json_str)
            else:
                overall_evaluation = json.loads(response_text)

            return overall_evaluation

        except json.JSONDecodeError as e:
            return {
                "error": "LLM 返回的 JSON 格式不正确",
                "raw_response": response_text,
                "parse_error": str(e)
            }

        except Exception as e:
            return {
                "error": "生成总体评价时发生错误",
                "exception": str(e)
            }

    async def run_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """运行决策（便捷方法）"""
        return await self.decide_next_step(state)

    async def run_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """运行最终评价（便捷方法）"""
        return await self.generate_final_evaluation(state)
