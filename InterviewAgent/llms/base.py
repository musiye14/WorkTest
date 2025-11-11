"""
LLM 基类
定义统一的 LLM 调用接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """
    LLM 基类，定义统一接口
    """

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        self.api_key = api_key
        self.model_name = model_name or self.get_default_model()
        self.temperature = temperature
        self.extra_params = kwargs

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        调用 LLM（同步）

        Args:
            prompt: 提示词
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            str: LLM 响应文本
        """
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """
        调用 LLM（异步）

        Args:
            prompt: 提示词
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            str: LLM 响应文本
        """
        pass

    @abstractmethod
    def invoke_with_schema(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        调用 LLM 并返回结构化输出（同步）

        Args:
            prompt: 提示词
            output_schema: JSON Schema 定义
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 解析后的 JSON 对象
        """
        pass

    @abstractmethod
    async def ainvoke_with_schema(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        调用 LLM 并返回结构化输出（异步）

        Args:
            prompt: 提示词
            output_schema: JSON Schema 定义
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 解析后的 JSON 对象
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_model(cls) -> str:
        """
        获取默认模型名称

        Returns:
            str: 默认模型名称
        """
        pass
