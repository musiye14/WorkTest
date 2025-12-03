"""
Chunker注册器模块

提供基于装饰器的Chunker注册机制和工厂方法
"""
from typing import Dict, Type, List, Optional
import os


class ChunkerRegistry:
    """Chunker注册器 - 管理所有文件类型的Chunker"""

    _registry: Dict[str, Type['ChunkerBase']] = {}

    @classmethod
    def register(cls, *extensions: str):
        """
        装饰器: 注册Chunker类到指定的文件扩展名

        用法:
            @ChunkerRegistry.register('pdf')
            class PDFChunk(ChunkerBase):
                pass

            @ChunkerRegistry.register('txt', 'md', 'text')
            class TxtChunker(ChunkerBase):
                pass

        参数:
            *extensions: 一个或多个文件扩展名(不含点号)
        """
        def decorator(chunker_class: Type['ChunkerBase']):
            if not extensions:
                raise ValueError(f"{chunker_class.__name__} 必须指定至少一个文件扩展名")

            for ext in extensions:
                ext_lower = ext.lower().lstrip('.')

                if ext_lower in cls._registry:
                    existing = cls._registry[ext_lower].__name__
                    raise TypeError(
                        f"扩展名 '{ext_lower}' 已被 {existing} 注册，"
                        f"无法重复注册到 {chunker_class.__name__}"
                    )

                cls._registry[ext_lower] = chunker_class
                print(f"[OK] 注册 {chunker_class.__name__} -> .{ext_lower}")

            return chunker_class

        return decorator

    @classmethod
    def create(cls, filepath: str, issemantic: bool = False) -> 'ChunkerBase':
        """
        工厂方法: 根据文件路径自动创建对应的Chunker实例

        参数:
            filepath: 文件路径
            issemantic: 是否使用语义分块

        返回:
            对应的Chunker实例

        异常:
            ValueError: 文件路径为空或无扩展名
            NotImplementedError: 不支持的文件类型
        """
        if not filepath:
            raise ValueError("文件路径不能为空")

        ext = os.path.splitext(filepath)[1].lower().lstrip('.')

        if not ext:
            raise ValueError(f"无法从路径 '{filepath}' 提取文件扩展名")

        chunker_class = cls._registry.get(ext)

        if chunker_class is None:
            supported = ', '.join(f'.{e}' for e in sorted(cls._registry.keys()))
            raise NotImplementedError(
                f"不支持的文件类型: .{ext}\n"
                f"当前支持的类型: {supported}"
            )

        return chunker_class(filepath, issemantic)

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取所有支持的文件扩展名列表"""
        return sorted(cls._registry.keys())

    @classmethod
    def is_supported(cls, filepath: str) -> bool:
        """检查文件类型是否支持"""
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        return ext in cls._registry

    @classmethod
    def get_chunker_class(cls, extension: str) -> Optional[Type['ChunkerBase']]:
        """根据扩展名获取Chunker类"""
        return cls._registry.get(extension.lower().lstrip('.'))
