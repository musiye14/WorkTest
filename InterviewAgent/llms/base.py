"""
LLM 基类
定义统一的 LLM 调用接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
from ..utils.logger import logger


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
        agent_name = kwargs.pop("agent_name", "InterviewAgent")

        start_time = time.time()
        result, usage = self._invoke_with_schema(prompt, output_schema, **kwargs)
        duration = time.time() - start_time

        # 使用 Loguru 记录日志
        node_logger = logger.bind(agent=agent_name, node=node_name)

        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = input_tokens + output_tokens

        # 同时保留控制台输出（用户友好）
        print(f"[埋点] 节点: {node_name} | "
              f"输入token: {input_tokens} | "
              f"输出token: {output_tokens} | "
              f"Total Tokens: {total_tokens} | "
              f"耗时: {duration:.2f}s")

        # 记录到日志文件
        node_logger.info(
            f"LLM调用完成 | 输入token: {input_tokens} | "
            f"输出token: {output_tokens} | 总计: {total_tokens} | "
            f"耗时: {duration:.2f}s"
        )

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
