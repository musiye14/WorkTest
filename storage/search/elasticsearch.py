"""
Elasticsearch 实现 - BM25 全文检索
"""
from typing import List, Optional
from elasticsearch import Elasticsearch
from .base import SearchEngineBase
from ..vector.base import Document, SearchResult
from config import get_config


class ElasticsearchStore(SearchEngineBase):
    """Elasticsearch 存储实现"""

    def __init__(
        self,
        index_name: str = "documents",
        host: str = None,
        port: int = None
    ):
        """
        初始化 Elasticsearch 连接

        参数:
            index_name: 索引名称
            host: ES 服务地址
            port: ES 服务端口
        """
        super().__init__(index_name)

        config = get_config()
        self.host = host if host is not None else config.get('ES_HOST', 'localhost')
        self.port = port if port is not None else config.get('ES_PORT', 9200)

        # 连接 ES
        self.client = Elasticsearch([f"http://{self.host}:{self.port}"])
        print(f"✓ 成功连接到 Elasticsearch: {self.host}:{self.port}")

    def create_index(self, drop_if_exists: bool = False) -> None:
        """
        创建索引

        Schema:
            - id: 文档 ID
            - content: 文本内容（IK 分词）
            - metadata: 元数据
        """
        if self.client.indices.exists(index=self.index_name):
            if drop_if_exists:
                self.client.indices.delete(index=self.index_name)
                print(f"✓ 删除已存在的索引: {self.index_name}")
            else:
                print(f"✓ 索引已存在: {self.index_name}")
                return

        # 定义索引映射
        mappings = {
            "properties": {
                "id": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "metadata": {"type": "object", "enabled": True}
            }
        }

        self.client.indices.create(index=self.index_name, mappings=mappings)
        print(f"✓ 创建索引: {self.index_name}")

    def insert(self, documents: List[Document]) -> List[str]:
        """插入文档到 ES"""
        if not documents:
            return []

        # 批量插入
        actions = []
        for doc in documents:
            actions.append({
                "index": {
                    "_index": self.index_name,
                    "_id": doc.id
                }
            })
            actions.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata or {}
            })

        if actions:
            self.client.bulk(operations=actions)
            print(f"✓ 插入 {len(documents)} 条文档到 ES")

        return [doc.id for doc in documents]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """BM25 全文检索"""
        # 构建查询
        query_body = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": top_k
        }

        # 添加过滤条件
        if filters:
            query_body["query"] = {
                "bool": {
                    "must": [{"match": {"content": query}}],
                    "filter": [
                        {"term": {f"metadata.{k}": v}} for k, v in filters.items()
                    ]
                }
            }

        # 执行搜索
        response = self.client.search(index=self.index_name, body=query_body)

        # 解析结果
        results = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                id=hit["_source"]["id"],
                content=hit["_source"]["content"],
                metadata=hit["_source"].get("metadata")
            )
            results.append(SearchResult(
                document=doc,
                score=hit["_score"],
                distance=0.0
            ))

        return results

    def delete(self, ids: List[str]) -> int:
        """删除文档"""
        if not ids:
            return 0

        # 批量删除
        actions = []
        for doc_id in ids:
            actions.append({"delete": {"_index": self.index_name, "_id": doc_id}})

        if actions:
            self.client.bulk(operations=actions)
            print(f"✓ 删除 {len(ids)} 条文档")

        return len(ids)

    def drop_index(self) -> None:
        """删除索引"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f"✓ 删除索引: {self.index_name}")
