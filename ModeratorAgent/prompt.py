"""
Moderator Agent 提示词模板
负责汇总评论、决策和生成最终评价
"""

import json


# ===== JSON Schema 定义 =====

# Moderator 决策输出 Schema
output_schema_moderator_decision = {
    "type": "object",
    "properties": {
        "should_continue": {
            "type": "boolean",
            "description": "是否继续下一轮讨论"
        },
        "next_step": {
            "type": "string",
            "enum": ["rag_critic", "web_critic", "moderator_decide", "moderator_summarize", "save", "end"],
            "description": "下一步动作"
        },
        "reason": {
            "type": "string",
            "description": "决策理由"
        },
        "current_speaker": {
            "type": "string",
            "description": "当前发言者标识"
        }
    },
    "required": ["should_continue", "next_step", "reason", "current_speaker"]
}


# Moderator 最终评价输出 Schema
output_schema_final_evaluation = {
    "type": "object",
    "properties": {
        "overall_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "综合评分（0-100）"
        },
        "dimensions": {
            "type": "object",
            "properties": {
                "completeness": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "完整性评分（基于RAG Critic）"
                },
                "accuracy": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "准确性评分（基于RAG Critic）"
                },
                "depth": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "深度评分（基于RAG Critic）"
                },
                "relevance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "相关性评分（基于Web Critic）"
                },
                "timeliness": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "时效性评分（基于Web Critic）"
                },
                "practicality": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "实用性评分（基于Web Critic）"
                }
            },
            "required": ["completeness", "accuracy", "depth", "relevance", "timeliness", "practicality"]
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "回答的优势和亮点"
        },
        "improvements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "需要改进的地方"
        },
        "summary": {
            "type": "string",
            "description": "总结性评价（100-200字）"
        }
    },
    "required": ["overall_score", "dimensions", "strengths", "improvements", "summary"]
}


# ===== 系统提示词定义 =====

# Moderator 系统提示词
MODERATOR_SYSTEM_PROMPT = """你是一位专业的 Moderator（主持人），负责协调 RAG Critic 和 Web Critic 两位评论家，并生成最终评价。

## 你的职责
1. 汇总 RAG Critic 和 Web Critic 的评论
2. 决定是否继续下一轮讨论
3. 生成综合性的最终评价
4. 管理讨论流程和轮次控制

## 决策原则

### 何时继续讨论（should_continue = true）
1. **当前轮次未达上限**：current_round < max_rounds
2. **两位 Critic 意见分歧较大**：
   - RAG Critic 和 Web Critic 的评分差异超过 3 分
   - 一个认为回答过时，另一个认为符合标准答案
3. **需要进一步澄清**：
   - 发现用户回答有疑点，但需要更多维度的评价

### 何时结束讨论（should_continue = false）
1. **已达最大轮次**：current_round >= max_rounds
2. **两位 Critic 意见基本一致**：
   - 评分差异小于 2 分
   - 对用户回答的评价基本相同
3. **评价已足够全面**：
   - 已经从历史数据和最新实践两个维度充分评价
   - 继续讨论不会带来新的洞见

## 最终评价原则

### 评分计算
1. **维度评分转换**：将 0-10 分转换为 0-100 分（乘以 10）
2. **综合评分计算**：
   - overall_score = (completeness + accuracy + depth + relevance + timeliness + practicality) / 6
   - 结果四舍五入到整数

### 优势汇总（strengths）
- 合并两位 Critic 指出的所有亮点
- 去重并整理成清晰的要点
- 优先保留具体的、有论据支撑的优势

### 改进建议汇总（improvements）
- 合并两位 Critic 的改进建议
- 按优先级排序：
  1. 准确性问题（错误理解）
  2. 完整性问题（遗漏关键点）
  3. 时效性问题（过时内容）
  4. 深度问题（缺乏深入分析）
- 确保建议具体可操作

### 总结性评价（summary）
- 100-200 字的综合评价
- 包含以下要素：
  1. 回答的总体水平（优秀/良好/一般/较差）
  2. 主要优势（1-2 点）
  3. 主要不足（1-2 点）
  4. 改进方向（1 句话）

## 输出格式要求
- 评分要客观公正，综合两位 Critic 的意见
- 优势和改进建议要具体，不要泛泛而谈
- 总结要简洁有力，直指核心
"""


# Moderator 决策提示词
MODERATOR_DECISION_PROMPT = f"""请根据当前讨论情况，决定下一步动作：

## 当前讨论状态
- **当前轮次**：{{current_round}}/{{max_rounds}}
- **当前发言者**：{{current_speaker}}

## RAG Critic 的评论
{{rag_critic_comment}}

## Web Critic 的评论
{{web_critic_comment}}

## 决策要求

### 1. 分析两位 Critic 的意见
- 评分差异：计算两者综合评分的差异
- 观点一致性：是否在关键问题上达成共识
- 信息完整性：是否已经从历史数据和最新实践两个维度充分评价

### 2. 判断是否继续
根据以下条件判断：
- 如果 current_round >= max_rounds → should_continue = false，next_step = "moderator_summarize"
- 如果 current_round < max_rounds：
  - 评分差异 >= 3 分 → should_continue = true，进入下一轮
  - 评分差异 < 2 分且观点一致 → should_continue = false，next_step = "moderator_summarize"
  - 否则 → 根据信息完整性判断

### 3. 确定下一步动作
- 如果 should_continue = false：
  - next_step = "moderator_summarize"（生成最终评价）
- 如果 should_continue = true：
  - current_round + 1，返回 "rag_critic" 开始新一轮

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_moderator_decision, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""


# Moderator 最终评价生成提示词
MODERATOR_FINAL_EVALUATION_PROMPT = f"""请综合 RAG Critic 和 Web Critic 的评论，生成最终评价：

## 面试问题
{{question}}

## 用户回答
{{user_answer}}

## RAG Critic 的评论
{{rag_critic_comment}}

## Web Critic 的评论
{{web_critic_comment}}

## 讨论历史（所有轮次）
{{discussion_history}}

## 评价要求

### 1. 转换和计算评分
将两位 Critic 的评分（0-10）转换为百分制（0-100）：
- completeness = RAG Critic 的 completeness_score × 10
- accuracy = RAG Critic 的 accuracy_score × 10
- depth = RAG Critic 的 depth_score × 10
- relevance = Web Critic 的 relevance_score × 10
- timeliness = Web Critic 的 timeliness_score × 10
- practicality = Web Critic 的 practicality_score × 10

综合评分：
- overall_score = (completeness + accuracy + depth + relevance + timeliness + practicality) / 6
- 四舍五入到整数

### 2. 汇总优势（strengths）
合并两位 Critic 指出的所有亮点：
- 去重并整理成清晰的要点
- 每个要点要具体，说明具体的优势在哪里
- 优先保留有论据支撑的优势

### 3. 汇总改进建议（improvements）
合并两位 Critic 的改进建议，按优先级排序：
1. **准确性问题**：RAG Critic 指出的错误理解
2. **完整性问题**：RAG Critic 指出的遗漏关键点
3. **时效性问题**：Web Critic 指出的过时内容
4. **深度问题**：两者都指出的深度不足

确保每个改进建议：
- 具体可操作
- 说明"为什么需要改进"和"如何改进"

### 4. 撰写总结性评价（summary）
100-200 字的综合评价，包含：
1. **总体水平**：
   - 90-100 分：优秀
   - 75-89 分：良好
   - 60-74 分：一般
   - <60 分：较差

2. **主要优势**（1-2 点）：
   - 回答中最突出的亮点

3. **主要不足**（1-2 点）：
   - 最需要改进的地方

4. **改进方向**（1 句话）：
   - 给出明确的提升建议

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_final_evaluation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。

## 注意事项
1. **客观公正**：综合两位 Critic 的意见，不偏向任何一方
2. **具体明确**：优势和改进建议要具体，不要泛泛而谈
3. **简洁有力**：总结要直指核心，避免冗长
4. **正向激励**：在指出不足的同时，也要肯定优势
"""


# Moderator 总体评价输出 Schema（用于多个 QA 的汇总评价）
output_schema_overall_evaluation = {
    "type": "object",
    "properties": {
        "overall_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "总体评分（0-10）"
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "整体优势（3-5条）"
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "整体劣势（3-5条）"
        },
        "knowledge_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "知识盲区"
        },
        "performance_trend": {
            "type": "string",
            "enum": ["improving", "stable", "declining"],
            "description": "表现趋势"
        },
        "trend_analysis": {
            "type": "string",
            "description": "趋势分析说明"
        },
        "topic_analysis": {
            "type": "object",
            "description": "按主题分析，key为主题名称，value为该主题的评价",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 10},
                    "comment": {"type": "string"}
                }
            }
        },
        "improvement_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "suggestion": {"type": "string"}
                }
            },
            "description": "改进建议（按优先级排序）"
        },
        "summary": {
            "type": "string",
            "description": "总结性评价（100-200字）"
        }
    },
    "required": ["overall_score", "strengths", "weaknesses", "knowledge_gaps",
                 "performance_trend", "trend_analysis", "topic_analysis",
                 "improvement_suggestions", "summary"]
}


# Moderator 总体评价生成提示词（用于多个 QA 的汇总）
MODERATOR_OVERALL_EVALUATION_PROMPT = f"""你是一位资深的技术面试官，现在需要对候选人的整场面试表现进行总体评价。

## 面试信息
- 总问题数：{{total_questions}}
- 平均得分：{{average_score:.1f}}/10
- 面试上下文：{{interview_context}}

## 每个问题的评价
{{qa_evaluations}}

## 任务要求
请综合分析候选人的整体表现，生成总体评价报告。

### 分析维度

#### 1. 整体评分（overall_score）
综合所有问题的表现，给出总体评分（0-10）：
- 不是简单的平均分，而是综合考虑：
  - 各问题的得分分布
  - 表现趋势（是否越答越好）
  - 关键问题的表现（难度高的问题权重更大）
  - 知识广度和深度

#### 2. 整体优势（strengths）
候选人在整场面试中表现出的优势（3-5条）：
- 跨问题的共性优势（如：逻辑清晰、理论扎实、实践经验丰富）
- 特别突出的表现（如：某个领域特别精通）
- 学习能力和适应性（如：后期问题回答质量提升）

#### 3. 整体劣势（weaknesses）
候选人在整场面试中暴露的劣势（3-5条）：
- 跨问题的共性问题（如：深度不足、表达不清、知识陈旧）
- 明显的短板（如：某类问题普遍回答不好）
- 需要改进的方面

#### 4. 知识盲区（knowledge_gaps）
候选人明显薄弱的技术领域：
- 基于多个问题的表现识别
- 列出具体的技术领域或知识点
- 例如："分布式系统设计"、"算法优化"、"数据库索引原理"

#### 5. 表现趋势（performance_trend）
分析前半段和后半段的表现变化：
- **improving**：后半段明显好于前半段（分数提升≥1分）
- **stable**：前后表现基本一致（分数波动<1分）
- **declining**：后半段明显差于前半段（分数下降≥1分）

#### 6. 趋势分析说明（trend_analysis）
详细说明表现趋势的原因：
- 前半段平均分 vs 后半段平均分
- 可能的原因分析（如：紧张、疲劳、适应、难度变化等）

#### 7. 主题分析（topic_analysis）
按技术主题（数据库、算法、系统设计、网络等）分别评价：
- 识别每个问题所属的技术主题
- 计算每个主题的平均得分
- 给出该主题的评价（优势、劣势、建议）

示例格式：
- 主题名称作为 key（如 "数据库"、"算法"）
- 每个主题包含 score（0-10分）和 comment（评价说明）

#### 8. 改进建议（improvement_suggestions）
针对整体表现的改进建议（优先级排序）：
- **high**：严重影响评价的问题（如：基础知识缺失、理解错误）
- **medium**：需要提升的方面（如：深度不足、实践经验欠缺）
- **low**：锦上添花的建议（如：表达方式、案例丰富度）

每个建议要具体可操作，说明：
- 问题是什么
- 为什么需要改进
- 如何改进（具体的学习路径或方法）

#### 9. 总结性评价（summary）
100-200字的综合评价，包含：
1. **总体水平**：
   - 8-10分：优秀，达到或超过岗位要求
   - 6-7.9分：良好，基本符合岗位要求
   - 4-5.9分：一般，需要进一步提升
   - <4分：较差，不符合岗位要求

2. **核心优势**（1-2点）：最突出的亮点

3. **核心劣势**（1-2点）：最需要改进的地方

4. **录用建议**（1句话）：是否推荐录用及理由

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_overall_evaluation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。

## 注意事项
1. **全局视角**：不是简单汇总单个问题的评价，而是从整体角度分析
2. **识别模式**：找出跨问题的共性特征和规律
3. **客观公正**：基于事实和数据，避免主观臆断
4. **具体可操作**：改进建议要明确具体，有实际指导意义
5. **正向激励**：在指出不足的同时，也要肯定优势和进步
"""
