"""
LLM 基类
定义统一的 LLM 调用接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time


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
    def _invoke_with_schema(self, prompt: str, output_schema: Dict[str, Any], **kwargs) -> tuple[Dict[str, Any], Dict]:
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

    def invoke_with_schema(self, prompt:str,
    output_schema:Dict[str,Any],**kwargs) ->Dict[str, Any]:
        node_name = kwargs.pop("node_name", "unknown")
        start_time = time.time()
        result, usage = self._invoke_with_schema(prompt, output_schema, **kwargs)
        duration = time.time() - start_time
        print(f"[埋点] 节点: {node_name} | "
              f"输入toekn: {usage.get('prompt_tokens', 0)} | "
              f"输出token: {usage.get('completion_tokens', 0)} | "
              f"Total Tokens: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)} | "
              f"耗时: {duration:.2f}s")
        return result


    @classmethod
    @abstractmethod
    def get_default_model(cls) -> str:
        """
        获取默认模型名称

        Returns:
            str: 默认模型名称
        """
        pass
