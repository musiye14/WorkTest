"""
OpenAI LLM 实现
支持 OpenAI API 和兼容的 API（如 DeepSeek）
"""

import json
from typing import Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
from .base import BaseLLM


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

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """调用 LLM（异步）"""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            **{k: v for k, v in kwargs.items() if k != "temperature"}
        )
        return response.choices[0].message.content

    def invoke_with_schema(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """调用 LLM 并返回结构化输出（同步）"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            response_format={"type": "json_object"},
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "output_schema"]}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    async def ainvoke_with_schema(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """调用 LLM 并返回结构化输出（异步）"""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            response_format={"type": "json_object"},
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "output_schema"]}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    @classmethod
    def get_default_model(cls) -> str:
        """获取默认模型名称"""
        return "gpt-3.5-turbo"
