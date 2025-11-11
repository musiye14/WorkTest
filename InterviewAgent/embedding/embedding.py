from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings
from typing import List

class YEmbedding(Embeddings):

    def __init__(self) -> None:
        super().__init__()
        self.model = BGEM3FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True,
                  cache_dir="./models")
    def embed_documents(self,texts: List[str]) -> List[List[float]]:
        outputs = self.model.encode(texts)
        return outputs["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        outputs = self.model.encode([text])
        return outputs["dense_vecs"][0].tolist()





