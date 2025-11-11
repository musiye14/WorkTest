"""
PostgreSQL 实现 - 主数据源
"""
from typing import List, Optional, Dict, Any
import asyncpg
from .base import DatabaseBase
from config import get_config
import uuid


class PostgreSQLDatabase(DatabaseBase):
    """PostgreSQL 数据库实现"""

    def __init__(self, database_url: str = None):
        """
        初始化 PostgreSQL 连接

        参数:
            database_url: 数据库连接 URL
        """
        config = get_config()
        self.database_url = database_url or config.get(
            'DATABASE_URL',
            'postgresql://user:password@localhost:5432/interview_db'
        )
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """连接数据库"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )
        print(f"✓ 成功连接到 PostgreSQL")

        # 创建表
        await self._create_tables()

    async def close(self) -> None:
        """关闭连接"""
        if self.pool:
            await self.pool.close()

    async def _create_tables(self) -> None:
        """创建表"""
        async with self.pool.acquire() as conn:
            # 文档表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    user_id UUID,
                    content TEXT NOT NULL,
                    doc_type VARCHAR(50),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # 语义记忆表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL,
                    topic VARCHAR(200) NOT NULL,
                    category VARCHAR(100),
                    proficiency INTEGER DEFAULT 0,
                    practice_count INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    last_practice TIMESTAMP,
                    first_learned TIMESTAMP DEFAULT NOW(),
                    weak_points TEXT[],
                    strong_points TEXT[],
                    status VARCHAR(20),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, topic)
                )
            """)

            # 情节记忆表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    abstract_question TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    topic VARCHAR(200),
                    user_context JSONB,
                    user_answer TEXT,
                    evaluation JSONB,
                    source VARCHAR(50),
                    company VARCHAR(100),
                    difficulty VARCHAR(20),
                    quality_score DECIMAL(3,1),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # 创建索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
                CREATE INDEX IF NOT EXISTS idx_semantic_memory_user_id ON semantic_memory(user_id);
                CREATE INDEX IF NOT EXISTS idx_semantic_memory_topic ON semantic_memory(topic);
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_topic ON episodic_memory(topic);
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_quality_score ON episodic_memory(quality_score);
            """)

            print("✓ 数据库表已创建")

    async def insert_document(self, document: Dict[str, Any]) -> str:
        """插入文档"""
        doc_id = document.get('id') or str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO documents (id, user_id, content, doc_type, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """, doc_id, document.get('user_id'), document['content'],
                document.get('doc_type'), document.get('metadata'))

        return doc_id

    async def insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """批量插入文档"""
        doc_ids = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for doc in documents:
                    doc_id = doc.get('id') or str(uuid.uuid4())
                    await conn.execute("""
                        INSERT INTO documents (id, user_id, content, doc_type, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                    """, doc_id, doc.get('user_id'), doc['content'],
                        doc.get('doc_type'), doc.get('metadata'))
                    doc_ids.append(doc_id)

        print(f"✓ 插入 {len(doc_ids)} 条文档到 PostgreSQL")
        return doc_ids

    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取文档"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", doc_id
            )

        if row:
            return dict(row)
        return None

    async def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取文档"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM documents WHERE id = ANY($1::uuid[])", doc_ids
            )

        return [dict(row) for row in rows]

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE id = $1", doc_id
            )

        return result == "DELETE 1"

    async def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """更新文档"""
        set_clause = ", ".join([f"{k} = ${i+2}" for i, k in enumerate(updates.keys())])
        query = f"UPDATE documents SET {set_clause}, updated_at = NOW() WHERE id = $1"

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, doc_id, *updates.values())

        return result == "UPDATE 1"

    async def get_semantic_memory_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取语义记忆"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM semantic_memory WHERE id = ANY($1::uuid[])", memory_ids
            )
        return [dict(row) for row in rows]

    async def get_episodic_memory_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取情节记忆"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM episodic_memory WHERE id = ANY($1::uuid[])", memory_ids
            )
        return [dict(row) for row in rows]
