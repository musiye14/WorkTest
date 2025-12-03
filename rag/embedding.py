from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings
from typing import List
import warnings


class YEmbedding(Embeddings):
    """
    基于 BGEM3 的中文向量嵌入模型

    特点：
    - 支持中文语义检索
    - 使用 FP16 加速推理
    - 自动缓存模型到本地
    """

    def __init__(self) -> None:
        super().__init__()

        # 忽略 tokenizer 的性能警告（这是 FlagEmbedding 内部实现的问题）
        warnings.filterwarnings('ignore', message='.*BertTokenizerFast.*')

        self.model = BGEM3FlagModel(
            'BAAI/bge-large-zh-v1.5',
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True,
            cache_dir="./models"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文档

        Args:
            texts: 文档列表

        Returns:
            向量列表
        """
        outputs = self.model.encode(texts)
        return outputs["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        编码单个查询

        Args:
            text: 查询文本

        Returns:
            向量
        """
        outputs = self.model.encode([text])
        return outputs["dense_vecs"][0].tolist()





