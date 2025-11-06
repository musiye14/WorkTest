"""
RAG 统一检索接口 - 混合检索（Milvus + ES + Rerank）

修正版：Milvus 只存索引，PostgreSQL 存完整数据
"""
from typing import List, Optional, Dict
from storage.manager import StorageManager
from storage.vector.base import Document, SearchResult
from .embedding import YEmbedding
from .reranker import Reranker


class RAG:
    """
    RAG 统一检索接口

    职责：
    1. 混合检索（Milvus 向量 + ES BM25）
    2. Rerank 重排序
    3. 文档处理（分块、嵌入、存储）

    架构：
    - Milvus/ES 只存索引（doc_id + embedding/text）
    - PostgreSQL 存完整数据
    - 检索后通过 doc_id 回查 PostgreSQL
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        embedding_model: Optional[YEmbedding] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        初始化 RAG

        参数:
            storage_manager: 存储管理器
            embedding_model: 嵌入模型（可选）
            reranker: 重排序器（可选）
        """
        self.storage = storage_manager
        self.embedding_model = embedding_model or YEmbedding()
        self.reranker = reranker or Reranker()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "rag_documents",
        filters: Optional[dict] = None,
        use_hybrid: bool = False
    ) -> List[SearchResult]:
        """
        统一检索接口

        参数:
            query: 查询文本
            top_k: 返回 top-k 个结果
            collection: Milvus 集合名称
            filters: 过滤条件
            use_hybrid: 是否使用混合检索（向量 + BM25）

        返回:
            搜索结果列表（包含完整文档内容）
        """
        if use_hybrid:
            return await self.hybrid_search(query, top_k, filters)
        else:
            return await self.vector_search(query, top_k, collection, filters)

    async def vector_search(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "rag_documents",
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        纯向量检索

        流程：
        1. Milvus 向量检索 → 返回 doc_ids
        2. PostgreSQL 批量查询 → 返回完整文档

        参数:
            query: 查询文本
            top_k: 返回 top-k 个结果
            collection: Milvus 集合名称
            filters: 过滤条件

        返回:
            搜索结果列表（包含完整文档内容）
        """
        # 1. 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)

        # 2. Milvus 向量检索（只返回 doc_ids）
        milvus = self.storage.get_milvus()
        milvus_results = milvus.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_expr=self._build_filter_expr(filters) if filters else None
        )

        # 3. 提取 doc_ids
        doc_ids = [r.document.id for r in milvus_results]

        if not doc_ids:
            return []

        # 4. 从 PostgreSQL 批量查询完整文档
        db = self.storage.get_db()
        documents = await db.get_documents_by_ids(doc_ids)

        # 5. 构建 id -> document 映射
        id_to_doc = {str(doc['id']): doc for doc in documents}

        # 6. 合并结果（保持 Milvus 的相似度排序）
        final_results = []
        for milvus_result in milvus_results:
            doc_id = milvus_result.document.id
            if doc_id in id_to_doc:
                pg_doc = id_to_doc[doc_id]
                # 创建完整的 Document
                full_doc = Document(
                    id=str(pg_doc['id']),
                    content=pg_doc['content'],
                    metadata=pg_doc.get('metadata')
                )
                # 保留 Milvus 的相似度分数
                final_results.append(SearchResult(
                    document=full_doc,
                    score=milvus_result.score,
                    distance=milvus_result.distance
                ))

        return final_results

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        混合检索（向量 + BM25 + Rerank）

        流程：
        1. Milvus 向量检索 (top_k * 2) → doc_ids
        2. ES BM25 检索 (top_k * 2) → doc_ids
        3. 合并去重 doc_ids
        4. PostgreSQL 批量查询完整文档
        5. Rerank 重排序
        6. 返回 top_k

        参数:
            query: 查询文本
            top_k: 返回 top-k 个结果
            filters: 过滤条件

        返回:
            搜索结果列表（包含完整文档内容）
        """
        # 1. 向量检索（已包含 PostgreSQL 回查）
        vector_results = await self.vector_search(
            query=query,
            top_k=top_k * 2,
            filters=filters
        )

        # 2. BM25 检索
        es = self.storage.get_es()
        bm25_results = es.search(
            query=query,
            top_k=top_k * 2,
            filters=filters
        )

        # 3. 提取 BM25 的 doc_ids
        bm25_doc_ids = [r.document.id for r in bm25_results]

        # 4. 从 PostgreSQL 查询 BM25 结果的完整文档
        if bm25_doc_ids:
            db = self.storage.get_db()
            bm25_documents = await db.get_documents_by_ids(bm25_doc_ids)
            id_to_doc = {str(doc['id']): doc for doc in bm25_documents}

            # 补全 BM25 结果的文档内容
            for result in bm25_results:
                if result.document.id in id_to_doc:
                    pg_doc = id_to_doc[result.document.id]
                    result.document.content = pg_doc['content']
                    result.document.metadata = pg_doc.get('metadata')

        # 5. 合并去重
        merged = self._merge_results(vector_results, bm25_results)

        # 6. Rerank 重排序
        reranked = self.reranker.rerank(query, merged, top_k=top_k)

        return reranked

    async def add_documents(
        self,
        documents: List[Document],
        collection: str = "rag_documents",
        generate_questions: bool = False
    ) -> List[str]:
        """
        添加文档（正确顺序：PostgreSQL → Milvus → ES）

        参数:
            documents: 文档列表
            collection: Milvus 集合名称
            generate_questions: 是否生成假设性问题

        返回:
            插入的文档 ID 列表
        """
        # 1. 写入 PostgreSQL（主数据源，获取 ID）
        db = self.storage.get_db()
        doc_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        doc_ids = await db.insert_documents(doc_dicts)

        # 更新 documents 的 ID
        for doc, doc_id in zip(documents, doc_ids):
            doc.id = doc_id

        # 2. 生成嵌入
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.embed_documents(contents)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        # 3. 写入 Milvus（向量索引）
        milvus = self.storage.get_milvus()
        milvus.insert(documents)

        # 4. 写入 ES（全文索引）
        es = self.storage.get_es()
        es.insert(documents)

        # 5. 生成假设性问题（可选）
        if generate_questions:
            await self._add_hypothetical_questions(documents)

        return doc_ids

    async def _add_hypothetical_questions(self, documents: List[Document]):
        """
        为文档生成假设性问题并存储到独立 Collection

        参数:
            documents: 文档列表
        """
        from .hypothetical_questions import HypotheticalQuestionGenerator

        generator = HypotheticalQuestionGenerator()
        db = self.storage.get_db()

        for doc in documents:
            # 1. 生成问题
            questions = generator.generate_questions(doc.content, num_questions=3)

            if not questions:
                continue

            # 2. 创建问题-文档映射
            mappings = generator.create_question_document_mapping(
                doc_id=doc.id,
                doc_content=doc.content,
                questions=questions
            )

            # 3. 将问题存储到 PostgreSQL
            question_dicts = [
                {
                    "id": m["id"],
                    "content": m["content"],
                    "metadata": m["metadata"]
                }
                for m in mappings
            ]
            question_ids = await db.insert_documents(question_dicts)

            # 4. 将问题存储到向量数据库
            question_docs = [
                Document(
                    id=qid,
                    content=m["content"],
                    metadata=m["metadata"]
                )
                for qid, m in zip(question_ids, mappings)
            ]

            # 生成问题的嵌入
            question_contents = [qd.content for qd in question_docs]
            question_embeddings = self.embedding_model.embed_documents(question_contents)

            for qd, emb in zip(question_docs, question_embeddings):
                qd.embedding = emb

            # 写入 Milvus（使用独立的 hypothetical_questions collection）
            # TODO: 需要初始化 hypothetical_questions collection
            milvus = self.storage.get_milvus()
            milvus.insert(question_docs)

            print(f"✓ 为文档 {doc.id} 生成了 {len(questions)} 个假设性问题")

    async def search_long_term_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """检索长期记忆"""
        return await self.search(
            query=query,
            top_k=top_k,
            collection="long_term_memory",
            filters={"user_id": user_id}
        )

    async def search_episodic_memory(
        self,
        query: str,
        user_context: Dict,
        top_k: int = 5
    ) -> List[SearchResult]:
        """检索事务记忆（Few-shot 案例）"""
        return await self.search(
            query=query,
            top_k=top_k,
            collection="episodic_memory",
            filters=user_context
        )

    async def search_standard_answers(
        self,
        query: str,
        top_k: int = 3
    ) -> List[SearchResult]:
        """检索标准答案库"""
        return await self.search(
            query=query,
            top_k=top_k,
            collection="standard_answers"
        )

    def _build_filter_expr(self, filters: dict) -> str:
        """构建 Milvus 过滤表达式"""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f"metadata['{key}'] == '{value}'")
            else:
                conditions.append(f"metadata['{key}'] == {value}")

        return " and ".join(conditions)

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult]
    ) -> List[SearchResult]:
        """合并向量检索和 BM25 检索结果"""
        merged_dict = {}

        for result in vector_results:
            doc_id = result.document.id
            merged_dict[doc_id] = result

        for result in bm25_results:
            doc_id = result.document.id
            if doc_id not in merged_dict or result.score > merged_dict[doc_id].score:
                merged_dict[doc_id] = result

        return list(merged_dict.values())
