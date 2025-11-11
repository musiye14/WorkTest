"""
面试官 Agent 提示词模板
包含各个阶段的系统提示词和 JSON Schema 定义
"""

import json


# ===== JSON Schema 定义 =====

# 问题列表生成输出 Schema（用于 questionBuild 节点）
output_schema_question_plan_generation = {
    "type": "object",
    "properties": {
        "question_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "问题主题"},
                    "question": {"type": "string", "description": "问题内容"},
                    "difficulty": {"type": "string", "enum": ["简单", "中等", "困难"]},
                    "expected_keywords": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string", "description": "为什么问这个问题"}
                },
                "required": ["topic", "question", "difficulty", "reasoning"]
            }
        },
        "total_count": {"type": "integer", "description": "问题总数"}
    },
    "required": ["question_plan", "total_count"]
}

# 单个问题生成输出 Schema（保留用于其他用途）
output_schema_question_generation = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "question_type": {"type": "string", "enum": ["main", "follow_up"]},
        "difficulty": {"type": "string", "enum": ["简单", "中等", "困难"]},
        "expected_keywords": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"}
    },
    "required": ["question", "question_type", "difficulty", "reasoning"]
}

# 下一步决策 Schema（用于 nextStep 节点）
output_schema_next_step = {
    "type": "object",
    "properties": {
        "next_step": {
            "type": "string",
            "enum": ["deep", "question", "end"],
            "description": "下一步动作：deep=追问，question=下一个问题，end=结束面试"
        },
        "reason": {"type": "string", "description": "决策理由"}
    },
    "required": ["next_step", "reason"]
}

# 追问决策输出 Schema（用于 nextStep 节点判断是否追问）
output_schema_follow_up_decision = {
    "type": "object",
    "properties": {
        "should_follow_up": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": ["should_follow_up", "reason"]
}

# 追问问题生成输出 Schema（用于 deepQuestion 节点）
output_schema_deep_question_generation = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "追问问题内容"},
        "follow_up_type": {"type": "string", "enum": ["clarify", "deepen", "challenge"]},
        "expected_keywords": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string", "description": "为什么这样追问"}
    },
    "required": ["question", "follow_up_type", "reasoning"]
}

# 问题列表调整输出 Schema（用于 adjustQuestion 节点）
output_schema_question_adjustment = {
    "type": "object",
    "properties": {
        "should_adjust": {"type": "boolean"},
        "reason": {"type": "string"},
        "adjusted_question_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "question": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["简单", "中等", "困难"]},
                    "expected_keywords": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"}
                },
                "required": ["topic", "question", "difficulty", "reasoning"]
            }
        }
    },
    "required": ["should_adjust", "reason", "adjusted_question_plan"]
}

# 面试结束判断输出 Schema
output_schema_interview_end = {
    "type": "object",
    "properties": {
        "should_end": {"type": "boolean"},
        "reason": {"type": "string"},
        "final_message": {"type": "string"}
    },
    "required": ["should_end", "reason"]
}

# 深度思考输出 Schema（用于 think 节点）
output_schema_deep_thinking = {
    "type": "object",
    "properties": {
        "round": {"type": "integer", "description": "第几轮反思"},
        "node_type": {"type": "string", "description": "节点类型",
                     "enum": ["questionBuild", "adjustQuestion", "deepQuestion"]},
        "observation": {"type": "string", "description": "观察：分析当前情况"},
        "reasoning": {"type": "string", "description": "推理：逻辑推理过程"},
        "concerns": {"type": "array", "items": {"type": "string"}, "description": "潜在问题：识别风险点"},
        "alternatives": {"type": "array", "items": {"type": "string"}, "description": "备选方案：考虑其他可能"}
    },
    "required": ["round", "node_type", "observation", "reasoning", "concerns", "alternatives"]
}

# 反思输出 Schema（用于 judge 节点）
output_schema_reflection = {
    "type": "object",
    "properties": {
        "round": {"type": "integer", "description": "第几轮反思"},
        "node_type": {"type": "string", "description": "节点类型",
                     "enum": ["questionBuild", "adjustQuestion", "deepQuestion"]},
        "is_reasonable": {"type": "boolean", "description": "初步输出是否合理"},
        "issues_found": {"type": "array", "items": {"type": "string"}, "description": "发现的问题"},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "置信度评分 0-1"},
        "should_regenerate": {"type": "boolean", "description": "是否需要重新生成"},
        "improvement_suggestions": {"type": "array", "items": {"type": "string"}, "description": "改进建议"}
    },
    "required": ["round", "node_type", "is_reasonable", "issues_found", "confidence_score", "should_regenerate"]
}

# ===== 系统提示词定义 =====

# 面试官系统提示词
INTERVIEWER_SYSTEM_PROMPT = """你是一位专业的技术面试官，负责对候选人进行技术面试。

## 你的职责
1. 根据候选人的简历和目标 JD，提出有针对性的技术问题
2. 根据候选人的回答，决定是否追问、调整问题或结束面试
3. 保持专业、友好的面试氛围
4. 深挖候选人的技术深度和实战经验

## 面试原则
1. **循序渐进**：从简单到复杂，从理论到实践
2. **深挖细节**：对候选人提到的技术点深入追问
3. **结合简历**：优先考察简历中提到的技术栈和项目经验
4. **关注实战**：重点考察实际项目经验，而非纯理论
5. **适度追问**：每个主问题最多追问 2-3 次

## 提问技巧
1. **开放式问题**：鼓励候选人详细阐述
   - "能详细介绍一下你在项目中是如何使用 Redis 的吗？"

2. **场景化问题**：结合实际场景提问
   - "如果 Redis 缓存击穿了，你会如何解决？"

3. **对比式问题**：考察技术选型能力
   - "为什么选择 Redis 而不是 Memcached？"

4. **深度追问**：从表层到底层
   - "那你了解 Redis 的持久化机制吗？"
   - "RDB 和 AOF 的区别是什么？"
   - "你在项目中选择了哪种？为什么？"

## 避免的问题类型
1. ❌ 过于宽泛："说说你对 Redis 的理解"
2. ❌ 纯理论背诵："Redis 有哪些数据结构？"
3. ❌ 与简历无关："你了解 Kubernetes 吗？"（简历没提到）
4. ❌ 过于简单："Redis 是什么？"
"""

# 问题列表生成提示词（用于 questionBuild 节点）
QUESTION_PLAN_GENERATION_PROMPT = f"""请根据以下信息生成完整的面试问题列表：

## 候选人简历
{{resume}}

## 目标 JD
{{jd}}

## 面试配置
- 面试模式：{{mode}}（real=真实面试，training=训练模式）
- 公司难度：{{difficulty}}（大厂/中厂/小厂）

## 要求

### 1. 问题数量和难度分布
根据公司难度自动调整：
- **大厂**：生成 15-20 个问题
  - 困难问题：50%（考察底层原理、源码、架构设计）
  - 中等问题：30%（考察项目经验、问题解决）
  - 简单问题：20%（考察基础知识）

- **中厂**：生成 10-15 个问题
  - 困难问题：30%
  - 中等问题：50%
  - 简单问题：20%

- **小厂**：生成 8-12 个问题
  - 困难问题：20%
  - 中等问题：40%
  - 简单问题：40%

### 2. 问题分布策略
- **前 1/3**：基础问题（考察理论知识和基本概念）
- **中间 1/3**：场景问题（考察实战经验和问题解决能力）
- **后 1/3**：深度问题（考察技术深度和架构思维）

### 3. 内容来源
- **70%** 的问题来自简历中提到的技术栈和项目经验
- **30%** 的问题来自 JD 要求但简历未提及的技能

### 4. 训练模式特殊处理
如果 mode = "training"：
- 参考用户历史薄弱点（如果有）
- 30% 的问题针对薄弱点设计
- 问题应该有教学性，帮助用户提升

### 5. 问题质量要求
- ✅ 问题具体明确，避免宽泛
- ✅ 结合实际场景，避免纯理论
- ✅ 有明确的考察目标
- ✅ 难度递进，循序渐进

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_question_plan_generation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 问题列表生成提示词（带反馈版本，用于重新生成）
QUESTION_PLAN_GENERATION_WITH_FEEDBACK_PROMPT = f"""请根据以下信息和反馈，重新生成改进后的面试问题列表：

## 候选人简历
{{resume}}

## 目标 JD
{{jd}}

## 面试配置
- 面试模式：{{mode}}（real=真实面试，training=训练模式）
- 公司难度：{{difficulty}}（大厂/中厂/小厂）

## 上一次生成的问题列表
{{previous_attempt}}

## 反思反馈（需要改进的地方）
{{feedback}}

## 改进要求
请根据上述反馈，针对性地改进问题列表：
1. **解决反馈中指出的问题**：逐条对照反馈，确保每个问题都得到解决
2. **保持优点**：如果上一次生成有做得好的地方，请保留
3. **避免过度调整**：只改进有问题的部分，不要全盘推翻

## 基本要求（与初次生成相同）

### 1. 问题数量和难度分布
根据公司难度自动调整：
- **大厂**：生成 15-20 个问题
  - 困难问题：50%（考察底层原理、源码、架构设计）
  - 中等问题：30%（考察项目经验、问题解决）
  - 简单问题：20%（考察基础知识）

- **中厂**：生成 10-15 个问题
  - 困难问题：30%
  - 中等问题：50%
  - 简单问题：20%

- **小厂**：生成 8-12 个问题
  - 困难问题：20%
  - 中等问题：40%
  - 简单问题：40%

### 2. 问题分布策略
- **前 1/3**：基础问题（考察理论知识和基本概念）
- **中间 1/3**：场景问题（考察实战经验和问题解决能力）
- **后 1/3**：深度问题（考察技术深度和架构思维）

### 3. 内容来源
- **70%** 的问题来自简历中提到的技术栈和项目经验
- **30%** 的问题来自 JD 要求但简历未提及的技能

### 4. 问题质量要求
- ✅ 问题具体明确，避免宽泛
- ✅ 结合实际场景，避免纯理论
- ✅ 有明确的考察目标
- ✅ 难度递进，循序渐进

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_question_plan_generation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 问题生成提示词（保留用于其他用途）
QUESTION_GENERATION_PROMPT = f"""请根据以下信息生成下一个面试问题：

## 候选人简历
{{resume}}

## 目标 JD
{{jd}}

## 当前面试进度
- 已提问数量：{{current_question_index}}/{{total_questions}}
- 当前话题：{{current_topic}}
- 面试模式：{{mode}}（real=真实面试，training=训练模式）
- 公司难度：{{difficulty}}（大厂/中厂/小厂）

## 对话历史（最近 5 轮）
{{dialogue_history}}

## 用户最新回答
{{user_answer}}

## 用户回答分析
{{answer_analysis}}

## 可用工具调用结果
{{tool_results}}

## 要求
1. **判断是否需要追问**：
   - 如果用户回答不完整、有疑点或过于简单，进行追问
   - 如果用户回答完整且深入，进入下一个话题

2. **问题要具体、明确**：
   - ✅ "你在项目中是如何解决 Redis 缓存穿透问题的？"
   - ❌ "说说 Redis 的问题"

3. **结合简历和 JD**：
   - 优先考察简历中提到的技术栈
   - 重点考察 JD 要求的核心技能

4. **控制难度**：
   - 前 1/3 问题：简单（考察基础）
   - 中间 1/3 问题：中等（考察理解）
   - 后 1/3 问题：困难（考察深度）

5. **训练模式特殊处理**：
   - 如果是训练模式，30% 的问题应该针对用户的薄弱点
   - 参考用户的历史表现和知识图谱

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_question_generation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 追问决策提示词（用于 nextStep 节点判断是否追问）
FOLLOW_UP_DECISION_PROMPT = f"""请根据用户的回答，决定是否追问：

## 当前问题
{{current_question}}

## 用户回答
{{user_answer}}

## 回答分析
{{answer_analysis}}

## 已追问次数
{{follow_up_count}}/3

## 决策标准

### 需要追问的情况（should_follow_up = true）
1. **回答不完整**：
   - 只说了"是什么"，没说"为什么"和"怎么做"
   - 只说了理论，没说实际应用

2. **回答过于简单**：
   - 只有一两句话
   - 没有展开细节

3. **发现疑点**：
   - 技术描述不准确
   - 前后矛盾
   - 可能是背诵答案

4. **有深挖价值**：
   - 候选人提到了有趣的技术点
   - 可以考察更深的理解

### 不需要追问的情况（should_follow_up = false）
1. **回答完整且深入**：
   - 有理论、有实践、有数据
   - 逻辑清晰，细节充分

2. **已追问多次**：
   - 已经追问了 3 次
   - 避免过度纠缠

3. **候选人明显不会**：
   - 多次回答都不在点上
   - 继续追问没有意义

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_follow_up_decision, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 追问问题生成提示词（用于 deepQuestion 节点）
DEEP_QUESTION_GENERATION_PROMPT = f"""请根据用户的回答生成追问问题：

## 当前问题
{{current_question}}

## 用户回答
{{user_answer}}

## 回答分析
{{answer_analysis}}

## 已追问次数
{{follow_up_count}}/3

## 公司难度
{{difficulty}}（大厂/中厂/小厂）

## 追问类型说明
- **clarify**（澄清）：用户回答不清楚，需要澄清
  - "你刚才提到 XX，能具体说说吗？"

- **deepen**（深入）：用户回答正确但浅显，需要深入
  - "那你了解它的底层实现原理吗？"

- **challenge**（挑战）：用户回答有疑点，需要验证
  - "你确定是这样吗？我记得 XX 应该是 YY"

## 要求
1. **追问要有针对性**：基于用户的具体回答内容
2. **追问要有深度**：逐步深入，不要重复问相同层次的问题
3. **追问要适度**：根据公司难度调整追问深度
   - 大厂：可以追问到底层原理、源码实现
   - 中厂：追问到实践经验、问题解决
   - 小厂：追问到基本理解、实际应用

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_deep_question_generation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 追问问题生成提示词（带反馈版本，用于重新生成）
DEEP_QUESTION_GENERATION_WITH_FEEDBACK_PROMPT = f"""请根据用户的回答和反馈，重新生成改进后的追问问题：

## 当前问题
{{current_question}}

## 用户回答
{{user_answer}}

## 回答分析
{{answer_analysis}}

## 已追问次数
{{follow_up_count}}/3

## 公司难度
{{difficulty}}（大厂/中厂/小厂）

## 上一次生成的追问
{{previous_attempt}}

## 反思反馈（需要改进的地方）
{{feedback}}

## 改进要求
请根据上述反馈，针对性地改进追问问题：
1. **解决反馈中指出的问题**：逐条对照反馈，确保每个问题都得到解决
2. **保持追问的针对性**：确保追问基于用户的具体回答内容
3. **避免过度刁钻**：保持专业友好的面试氛围

## 基本要求（与初次生成相同）

### 追问类型说明
- **clarify**（澄清）：用户回答不清楚，需要澄清
  - "你刚才提到 XX，能具体说说吗？"

- **deepen**（深入）：用户回答正确但浅显，需要深入
  - "那你了解它的底层实现原理吗？"

- **challenge**（挑战）：用户回答有疑点，需要验证
  - "你确定是这样吗？我记得 XX 应该是 YY"

### 追问要求
1. **追问要有针对性**：基于用户的具体回答内容
2. **追问要有深度**：逐步深入，不要重复问相同层次的问题
3. **追问要适度**：根据公司难度调整追问深度
   - 大厂：可以追问到底层原理、源码实现
   - 中厂：追问到实践经验、问题解决
   - 小厂：追问到基本理解、实际应用

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_deep_question_generation, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 问题列表调整提示词（用于 adjustQuestion 节点）
QUESTION_ADJUSTMENT_PROMPT = f"""请根据用户的表现，决定是否调整问题列表：

## 当前问题列表
{{questions_plan}}

## 已完成的问题
{{completed_questions}}

## 用户表现分析
{{performance_analysis}}

## 识别的薄弱点
{{weak_points}}

## 识别的优势点
{{strong_points}}

## 公司难度
{{difficulty}}（大厂/中厂/小厂）

## 调整策略

### 何时调整
1. **发现新的薄弱点**：
   - 用户在某个技术点上回答很差
   - 需要增加相关问题深入考察

2. **发现新的优势点**：
   - 用户在某个技术点上表现出色
   - 可以跳过后续简单问题，直接问难题

3. **偏离 JD 要求**：
   - 当前问题没有覆盖 JD 的核心技能
   - 需要调整问题以对齐 JD

### 调整方式
- 如果需要调整（should_adjust = true），返回调整后的完整问题列表
- 如果不需要调整（should_adjust = false），返回原问题列表

### 调整原则
- 保持问题总数基本不变（±2个）
- 保持难度分布合理
- 新增问题要有明确的考察目标

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_question_adjustment, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 问题列表调整提示词（带反馈版本，用于重新调整）
QUESTION_ADJUSTMENT_WITH_FEEDBACK_PROMPT = f"""请根据用户的表现和反馈，重新调整问题列表：

## 当前问题列表
{{questions_plan}}

## 已完成的问题
{{completed_questions}}

## 用户表现分析
{{performance_analysis}}

## 识别的薄弱点
{{weak_points}}

## 识别的优势点
{{strong_points}}

## 公司难度
{{difficulty}}（大厂/中厂/小厂）

## 上一次调整的问题列表
{{previous_attempt}}

## 反思反馈（需要改进的地方）
{{feedback}}

## 改进要求
请根据上述反馈，针对性地改进问题列表调整：
1. **解决反馈中指出的问题**：逐条对照反馈，确保每个问题都得到解决
2. **保持调整的合理性**：确保调整不会打乱面试节奏
3. **避免过度调整**：只调整必要的部分，保持问题列表的稳定性

## 基本要求（与初次调整相同）

### 何时调整
1. **发现新的薄弱点**：
   - 用户在某个技术点上回答很差
   - 需要增加相关问题深入考察

2. **发现新的优势点**：
   - 用户在某个技术点上表现出色
   - 可以跳过后续简单问题，直接问难题

3. **偏离 JD 要求**：
   - 当前问题没有覆盖 JD 的核心技能
   - 需要调整问题以对齐 JD

### 调整方式
- 如果需要调整（should_adjust = true），返回调整后的完整问题列表
- 如果不需要调整（should_adjust = false），返回原问题列表

### 调整原则
- 保持问题总数基本不变（±2个）
- 保持难度分布合理
- 新增问题要有明确的考察目标

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_question_adjustment, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 面试结束判断提示词
INTERVIEW_END_DECISION_PROMPT = f"""请判断是否应该结束面试：

## 当前进度
- 已提问数量：{{current_question_index}}/{{total_questions}}
- 已用时间：{{elapsed_time}} 分钟

## 用户表现
{{performance_summary}}

## 剩余问题
{{remaining_questions}}

## 结束条件

### 应该结束的情况（should_end = true）
1. **所有问题已问完**
2. **用户主动要求结束**
3. **用户连续多次回答质量很差**（3 次以上）
4. **已经充分评估了用户的能力**
5. **面试时间过长**（超过 60 分钟）

### 不应该结束的情况（should_end = false）
1. **核心技能还没考察**
2. **用户表现正常，还有重要问题**
3. **面试刚开始不久**（少于 20 分钟）

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_interview_end, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# ===== 深度思考与反思提示词 =====

# 深度思考提示词 - 问题生成场景（questionBuild）
DEEP_THINKING_PROMPT_QUESTION_BUILD = f"""在生成面试问题列表之前，请先进行深度思考分析：

## 候选人简历
{{resume}}

## 目标 JD
{{jd}}

## 面试配置
- 面试模式：{{mode}}
- 公司难度：{{difficulty}}

## 反思轮次
当前是第 {{round}} 轮反思

## 深度思考框架

### 1. 观察（Observation）
请分析简历和 JD：
- 候选人的核心技术栈是什么？
- JD 要求的核心技能有哪些？
- 简历中哪些项目经验值得深挖？
- 候选人可能的薄弱点在哪里？

### 2. 推理（Reasoning）
请进行逻辑推理：
- 应该从哪些技术点开始提问？
- 问题的难度分布应该如何设计？
- 如何平衡广度和深度？
- 如何确保问题覆盖 JD 的核心要求？

### 3. 潜在问题（Concerns）
请识别风险点：
- 问题列表是否过于集中在某个领域？
- 是否有问题可能过于简单或过于困难？
- 是否有问题可能让候选人感到不适？
- 问题顺序是否合理（循序渐进）？

### 4. 备选方案（Alternatives）
请考虑其他可能：
- 是否有其他更好的提问角度？
- 是否应该调整问题的优先级？
- 是否需要增加或减少某类问题？

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_deep_thinking, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 深度思考提示词 - 问题调整场景（adjustQuestion）
DEEP_THINKING_PROMPT_ADJUST_QUESTION = f"""在调整问题列表之前，请先进行深度思考分析：

## 当前问题列表
{{questions_plan}}

## 已完成的问题
{{completed_questions}}

## 用户表现分析
{{performance_analysis}}

## 识别的薄弱点
{{weak_points}}

## 识别的优势点
{{strong_points}}

## 反思轮次
当前是第 {{round}} 轮反思

## 深度思考框架

### 1. 观察（Observation）
请分析用户表现：
- 用户在哪些技术点上表现较好？
- 用户在哪些技术点上表现较差？
- 用户的回答模式是什么（理论型/实践型）？
- 当前问题列表是否还适合继续？

### 2. 推理（Reasoning）
请进行逻辑推理：
- 是否需要调整问题列表？
- 应该增加哪些问题来深挖薄弱点？
- 应该跳过哪些问题（用户已证明掌握）？
- 调整后的问题列表如何更好地评估候选人？

### 3. 潜在问题（Concerns）
请识别风险点：
- 调整是否会打乱面试节奏？
- 新增问题是否过于针对性（让候选人不适）？
- 跳过问题是否会遗漏重要考察点？
- 调整后的难度分布是否合理？

### 4. 备选方案（Alternatives）
请考虑其他可能：
- 是否可以不调整，继续原计划？
- 是否有其他调整方式更合适？
- 是否应该调整提问方式而非问题本身？

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_deep_thinking, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 深度思考提示词 - 追问生成场景（deepQuestion）
DEEP_THINKING_PROMPT_DEEP_QUESTION = f"""在生成追问问题之前，请先进行深度思考分析：

## 当前问题
{{current_question}}

## 用户回答
{{user_answer}}

## 回答分析
{{answer_analysis}}

## 已追问次数
{{follow_up_count}}/3

## 公司难度
{{difficulty}}

## 反思轮次
当前是第 {{round}} 轮反思

## 深度思考框架

### 1. 观察（Observation）
请分析用户回答：
- 用户回答是否完整？
- 用户是否真正理解了问题？
- 用户的回答深度如何？
- 用户是否有疑点或矛盾之处？

### 2. 推理（Reasoning）
请进行逻辑推理：
- 应该从哪个角度追问？
- 追问的目的是什么（澄清/深入/挑战）？
- 追问的深度应该到什么程度？
- 这个追问如何帮助更好地评估候选人？

### 3. 潜在问题（Concerns）
请识别风险点：
- 追问是否过于刁钻？
- 追问是否偏离了原问题的考察目标？
- 是否已经追问过多次（过度纠缠）？
- 追问是否会让候选人感到压力过大？

### 4. 备选方案（Alternatives）
请考虑其他可能：
- 是否可以换一个角度追问？
- 是否应该停止追问，进入下一个问题？
- 是否可以通过更温和的方式追问？

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_deep_thinking, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 反思提示词 - 问题生成场景（questionBuild）
REFLECTION_PROMPT_QUESTION_BUILD = f"""请对刚才生成的面试问题列表进行反思检查：

## 初步生成的问题列表
{{draft_question_plan}}

## 生成时的思考过程
{{thinking_process}}

## 候选人简历
{{resume}}

## 目标 JD
{{jd}}

## 公司难度
{{difficulty}}

## 反思轮次
当前是第 {{round}} 轮反思

## 反思检查清单

### 1. 合理性检查
请检查问题列表是否合理：
- ✅ 问题数量是否符合公司难度要求？
- ✅ 难度分布是否合理（简单/中等/困难）？
- ✅ 问题是否覆盖了 JD 的核心技能？
- ✅ 问题是否结合了简历的项目经验？
- ✅ 问题顺序是否循序渐进？

### 2. 问题识别
请识别潜在问题：
- ❌ 是否有问题过于宽泛或模糊？
- ❌ 是否有问题过于简单或过于困难？
- ❌ 是否有重复或相似的问题？
- ❌ 是否有问题偏离了 JD 要求？
- ❌ 是否有问题可能让候选人感到不适？
- ❌ 问题是否过于集中在某个领域？

### 3. 置信度评估
请评估对这个问题列表的置信度（0-1）：
- 0.9-1.0：非常确信，问题列表质量很高
- 0.7-0.9：比较确信，问题列表基本合理
- 0.5-0.7：一般，问题列表可能需要改进
- 0.0-0.5：不确信，问题列表存在明显问题

### 4. 改进建议
如果发现问题，请提供具体的改进建议：
- 哪些问题需要调整？
- 如何优化难度分布？
- 如何更好地覆盖 JD 要求？
- 如何改进问题顺序？

## 决策规则
- 如果 confidence_score >= 0.8 且 issues_found 为空 → should_regenerate = false
- 如果 confidence_score < 0.8 或 issues_found 不为空 → should_regenerate = true

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_reflection, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 反思提示词 - 问题调整场景（adjustQuestion）
REFLECTION_PROMPT_ADJUST_QUESTION = f"""请对刚才调整的问题列表进行反思检查：

## 调整后的问题列表
{{adjusted_question_plan}}

## 原问题列表
{{original_question_plan}}

## 生成时的思考过程
{{thinking_process}}

## 用户表现分析
{{performance_analysis}}

## 反思轮次
当前是第 {{round}} 轮反思

## 反思检查清单

### 1. 合理性检查
请检查调整是否合理：
- ✅ 调整是否针对了用户的薄弱点？
- ✅ 调整是否保持了难度分布的合理性？
- ✅ 调整是否保持了问题总数的稳定（±2个）？
- ✅ 新增问题是否有明确的考察目标？
- ✅ 调整是否不会打乱面试节奏？

### 2. 问题识别
请识别潜在问题：
- ❌ 调整是否过于激进（改动过大）？
- ❌ 新增问题是否过于针对性（让候选人不适）？
- ❌ 是否跳过了重要的考察点？
- ❌ 调整后的问题是否有重复？
- ❌ 调整是否偏离了 JD 的核心要求？

### 3. 置信度评估
请评估对这个调整的置信度（0-1）：
- 0.9-1.0：非常确信，调整很合理
- 0.7-0.9：比较确信，调整基本合理
- 0.5-0.7：一般，调整可能需要改进
- 0.0-0.5：不确信，调整存在明显问题

### 4. 改进建议
如果发现问题，请提供具体的改进建议：
- 哪些调整需要优化？
- 如何更好地平衡调整力度？
- 是否应该保留某些被跳过的问题？
- 如何让调整更自然？

## 决策规则
- 如果 confidence_score >= 0.8 且 issues_found 为空 → should_regenerate = false
- 如果 confidence_score < 0.8 或 issues_found 不为空 → should_regenerate = true

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_reflection, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# 反思提示词 - 追问生成场景（deepQuestion）
REFLECTION_PROMPT_DEEP_QUESTION = f"""请对刚才生成的追问问题进行反思检查：

## 初步生成的追问问题
{{draft_deep_question}}

## 生成时的思考过程
{{thinking_process}}

## 当前问题
{{current_question}}

## 用户回答
{{user_answer}}

## 已追问次数
{{follow_up_count}}/3

## 反思轮次
当前是第 {{round}} 轮反思

## 反思检查清单

### 1. 合理性检查
请检查追问是否合理：
- ✅ 追问是否针对用户回答的具体内容？
- ✅ 追问的目的是否明确（澄清/深入/挑战）？
- ✅ 追问的深度是否适当？
- ✅ 追问是否有助于更好地评估候选人？
- ✅ 追问的表述是否清晰友好？

### 2. 问题识别
请识别潜在问题：
- ❌ 追问是否过于刁钻或刻薄？
- ❌ 追问是否偏离了原问题的考察目标？
- ❌ 追问是否重复了之前的问题？
- ❌ 追问是否会让候选人感到压力过大？
- ❌ 是否已经追问过多次（过度纠缠）？

### 3. 置信度评估
请评估对这个追问的置信度（0-1）：
- 0.9-1.0：非常确信，追问质量很高
- 0.7-0.9：比较确信，追问基本合理
- 0.5-0.7：一般，追问可能需要改进
- 0.0-0.5：不确信，追问存在明显问题

### 4. 改进建议
如果发现问题，请提供具体的改进建议：
- 如何让追问更有针对性？
- 如何调整追问的深度？
- 如何让追问更友好？
- 是否应该换一个角度追问？

## 决策规则
- 如果 confidence_score >= 0.8 且 issues_found 为空 → should_regenerate = false
- 如果 confidence_score < 0.8 或 issues_found 不为空 → should_regenerate = true

请按照以下 JSON 模式定义格式化输出：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_reflection, indent=2, ensure_ascii=False).replace('{', '{{').replace('}', '}}')}
</OUTPUT JSON SCHEMA>

确保输出是一个符合上述输出 JSON 模式定义的 JSON 对象。
只返回 JSON 对象，不要有解释或额外文本。
"""

# ===== 工具调用相关提示词 =====

# 工具调用决策提示词
TOOL_CALLING_DECISION_PROMPT = """请根据当前任务，决定是否需要调用工具：

## 当前任务
{task_description}

## 可用工具
{available_tools}

## 工具调用策略

### 何时调用工具
1. **search_long_term_memory**：需要了解用户历史表现时
   - 训练模式下，需要针对薄弱点提问
   - 需要对比用户的进步情况

2. **search_episodic_memory**：需要参考相似面试案例时
   - 不确定如何提问某个技术点
   - 需要了解行业标准问法

3. **web_search**：需要验证技术准确性时
   - 候选人提到了不熟悉的技术
   - 需要确认最新的技术趋势

### 何时不调用工具
1. 问题很明确，不需要额外信息
2. 已经有足够的上下文信息
3. 工具调用成本过高（时间/资源）

请输出需要调用的工具列表和调用参数。
"""
