"""
OpenAI LLM 实现
支持 OpenAI API 和兼容的 API（如 DeepSeek）
"""

import json
from typing import Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
from .base import BaseLLM
from jsonschema import validate, ValidationError


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM 实现
    支持通过 base_url 使用兼容 OpenAI API 的服务
    """

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(api_key, model_name, temperature, **kwargs)

        # 初始化同步客户端
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # 初始化异步客户端
        self.async_client = AsyncOpenAI(**client_kwargs)

    def invoke(self, prompt: str, **kwargs) -> str:
        """调用 LLM（同步）"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            **{k: v for k, v in kwargs.items() if k != "temperature"}
        )
        return response.choices[0].message.content

    async def ainvoke(self, messages, **kwargs):
        """
        调用 LLM（异步）- 兼容 LangChain 接口

        Args:
            messages: LangChain 格式的消息列表（SystemMessage, HumanMessage等）
            **kwargs: 其他参数

        Returns:
            包含 content 和 usage 属性的响应对象
        """
        # 将 LangChain 格式的 messages 转换为 OpenAI 格式
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain Message 对象
                role = 'system' if msg.type == 'system' else 'user' if msg.type == 'human' else 'assistant'
                openai_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                # 字典格式
                openai_messages.append(msg)
            else:
                # 字符串格式
                openai_messages.append({"role": "user", "content": str(msg)})

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=kwargs.get("temperature", self.temperature),
            **{k: v for k, v in kwargs.items() if k != "temperature"}
        )

        # 提取 token 使用量
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

        # 返回类似 LangChain 的响应对象，同时包含 usage 信息
        class Response:
            def __init__(self, content, usage):
                self.content = content
                self.usage = usage

        return Response(response.choices[0].message.content, usage)

    def _invoke_with_schema(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> tuple[Dict[str, Any], Dict]:
        """调用 LLM 并返回结构化输出（同步）"""
        max_retries = 3
        last_result = None
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        for attempt in range(max_retries):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                response_format={"type": "json_object"},
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "output_schema"]}
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            last_result = result

            # 累计 token 使用量
            if hasattr(response, 'usage') and response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage["completion_tokens"] += response.usage.completion_tokens

            # 验证输出是否符合 schema
            try:
                validate(instance=result, schema=output_schema)
                # 验证成功，返回结果
                return result, total_usage
            except ValidationError as e:
                print(f"[警告] 第 {attempt + 1}/{max_retries} 次尝试，LLM 输出不符合 schema: {e.message}")
                if attempt < max_retries - 1:
                    print(f"[重试] 正在进行第 {attempt + 2} 次尝试...")
                else:
                    # 最后一次尝试仍然失败
                    print(f"[错误] 已达到最大重试次数 ({max_retries})，返回最后一次结果（可能不符合 schema）")
                    return last_result, total_usage

        # 理论上不会到这里，但为了安全返回最后的结果
        return last_result, total_usage

    @classmethod
    def get_default_model(cls) -> str:
        """获取默认模型名称"""
        return "gpt-3.5-turbo"
