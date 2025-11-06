"""
Milvus向量数据库实现
"""
from typing import List, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from .base import VectorStoreBase, Document, SearchResult
import uuid
from config import get_config


class MilvusStore(VectorStoreBase):
    """Milvus向量存储实现"""

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int = 1024,
        host: str = None,
        port: int = None,
        alias: str = "default"
    ):
        """
        初始化Milvus连接

        参数:
            collection_name: 集合名称
            embedding_dim: 向量维度
            host: Milvus服务地址
            port: Milvus服务端口
            alias: 连接别名
        """
        super().__init__(collection_name, embedding_dim)

        # 从配置读取,如果没有传参
        config = get_config()
        self.host = host if host is not None else config.get('MILVUS_HOST', 'localhost')
        self.port = port if port is not None else config.get('MILVUS_PORT', 19530)
        self.alias = alias
        self.collection: Optional[Collection] = None

        # 连接Milvus
        self._connect()

    def _connect(self):
        """连接到Milvus服务"""
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port
            )
            print(f"✓ 成功连接到Milvus: {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"无法连接到Milvus: {e}")

    def create_collection(self, drop_if_exists: bool = False) -> None:
        """
        创建Milvus集合

        Schema（只存索引，不存原文）:
            - id: 主键(VARCHAR) - 对应 PostgreSQL 的文档 ID
            - embedding: 向量(FLOAT_VECTOR)
            - metadata: 少量元数据(JSON) - 用于过滤，不存完整内容
        """
        # 检查集合是否存在
        if utility.has_collection(self.collection_name):
            if drop_if_exists:
                utility.drop_collection(self.collection_name)
                print(f"✓ 删除已存在的集合: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                print(f"✓ 加载已存在的集合: {self.collection_name}")
                return

        # 定义Schema（移除 content 字段）
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)  # 只存少量元数据
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"RAG向量索引: {self.collection_name}"
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.alias
        )

        # 创建索引(IVF_FLAT + COSINE相似度)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print(f"✓ 创建集合: {self.collection_name} (维度: {self.embedding_dim})")

    def insert(self, documents: List[Document]) -> List[str]:
        """插入向量索引到Milvus（不存原文）"""
        if not self.collection:
            raise RuntimeError("集合未初始化,请先调用create_collection()")

        if not documents:
            return []

        # 准备数据（只存 id + embedding + 少量 metadata）
        ids = []
        embeddings = []
        metadatas = []

        for doc in documents:
            if not doc.embedding:
                raise ValueError(f"文档 {doc.id} 缺少embedding")

            ids.append(doc.id or str(uuid.uuid4()))
            embeddings.append(doc.embedding)

            # 只存少量元数据（用于过滤）
            metadata = doc.metadata or {}
            # 移除大字段，只保留过滤字段
            filter_metadata = {
                k: v for k, v in metadata.items()
                if k in ['user_id', 'doc_type', 'category', 'type', 'original_doc_id']
            }
            metadatas.append(filter_metadata)

        # 插入数据（移除 content）
        data = [ids, embeddings, metadatas]
        self.collection.insert(data)
        self.collection.flush()

        print(f"✓ 插入 {len(documents)} 条向量索引到 {self.collection_name}")
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """向量相似度搜索（只返回 doc_id，不返回原文）"""
        if not self.collection:
            raise RuntimeError("集合未初始化")

        # 加载集合到内存
        self.collection.load()

        # 搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        # 执行搜索（只输出 id 和 metadata）
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "metadata"]  # 移除 content
        )

        # 解析结果（只包含 doc_id，不包含原文）
        search_results = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    id=hit.entity.get("id"),
                    content="",  # 不存原文，需要从 PostgreSQL 查询
                    metadata=hit.entity.get("metadata")
                )
                search_results.append(SearchResult(
                    document=doc,
                    score=hit.score,
                    distance=hit.distance
                ))

        return search_results

    def delete(self, ids: List[str]) -> int:
        """删除文档"""
        if not self.collection:
            raise RuntimeError("集合未初始化")

        expr = f"id in {ids}"
        self.collection.delete(expr)
        self.collection.flush()

        print(f"✓ 删除 {len(ids)} 条文档")
        return len(ids)

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        if not self.collection:
            raise RuntimeError("集合未初始化")

        self.collection.load()

        expr = f"id in {ids}"
        results = self.collection.query(
            expr=expr,
            output_fields=["id", "content", "metadata"]
        )

        documents = []
        for result in results:
            doc = Document(
                id=result["id"],
                content=result["content"],
                metadata=result.get("metadata")
            )
            documents.append(doc)

        return documents

    def count(self) -> int:
        """获取文档总数"""
        if not self.collection:
            raise RuntimeError("集合未初始化")

        return self.collection.num_entities

    def drop_collection(self) -> None:
        """删除集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"✓ 删除集合: {self.collection_name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """断开连接"""
        connections.disconnect(self.alias)
