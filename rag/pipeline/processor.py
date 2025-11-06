"""
文档处理器 - 整合分块、嵌入、存储的完整流水线
"""
from typing import List, Optional
from pathlib import Path
import uuid

from ..chunker import ChunkerRegistry
from ..embedding import YEmbedding
from storage.vector import VectorStoreBase, Document


class DocumentProcessor:
    """
    文档处理器 - RAG流水线核心类

    功能:
        1. 加载文档并分块
        2. 生成向量嵌入
        3. 存储到向量数据库
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_model: Optional[YEmbedding] = None
    ):
        """
        初始化文档处理器

        参数:
            vector_store: 向量存储实例
            embedding_model: 嵌入模型(可选,默认使用YEmbedding)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or YEmbedding()

    def process_file(
        self,
        filepath: str,
        issemantic: bool = False,
        metadata: Optional[dict] = None
    ) -> List[str]:
        """
        处理单个文件

        参数:
            filepath: 文件路径
            issemantic: 是否使用语义分块
            metadata: 额外的元数据

        返回:
            插入的文档ID列表
        """
        print(f"\n{'='*60}")
        print(f"处理文件: {filepath}")
        print(f"{'='*60}")

        # 1. 检查文件类型
        if not ChunkerRegistry.is_supported(filepath):
            supported = ChunkerRegistry.get_supported_extensions()
            raise ValueError(
                f"不支持的文件类型: {filepath}\n"
                f"支持的类型: {', '.join(f'.{ext}' for ext in supported)}"
            )

        # 2. 分块
        print(f"\n[1/3] 文档分块...")
        chunker = ChunkerRegistry.create(filepath, issemantic)
        chunks = list(chunker.chunker())
        print(f"✓ 分块完成: {len(chunks)} 个chunk")

        # 3. 生成嵌入
        print(f"\n[2/3] 生成向量嵌入...")
        documents = []
        chunk_texts = [self._extract_text(chunk) for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(chunk_texts)

        # 4. 构建Document对象
        file_path = Path(filepath)
        base_metadata = {
            "filename": file_path.name,
            "filepath": str(file_path.absolute()),
            "file_type": file_path.suffix[1:],
            **(metadata or {})
        }

        for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            doc = Document(
                id=str(uuid.uuid4()),
                content=chunk_text,
                embedding=embedding,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

        print(f"✓ 嵌入完成: {len(documents)} 个向量")

        # 5. 存储到向量数据库
        print(f"\n[3/3] 存储到向量数据库...")
        doc_ids = self.vector_store.insert(documents)
        print(f"✓ 存储完成: {len(doc_ids)} 条记录")

        print(f"\n{'='*60}")
        print(f"✓ 文件处理完成: {filepath}")
        print(f"{'='*60}\n")

        return doc_ids

    def process_directory(
        self,
        directory: str,
        issemantic: bool = False,
        recursive: bool = True
    ) -> dict:
        """
        批量处理目录下的所有文件

        参数:
            directory: 目录路径
            issemantic: 是否使用语义分块
            recursive: 是否递归处理子目录

        返回:
            处理结果字典 {filepath: doc_ids}
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"不是有效的目录: {directory}")

        # 获取支持的扩展名
        supported_exts = ChunkerRegistry.get_supported_extensions()

        # 查找文件
        pattern = "**/*" if recursive else "*"
        files = []
        for ext in supported_exts:
            files.extend(dir_path.glob(f"{pattern}.{ext}"))

        print(f"\n找到 {len(files)} 个文件待处理")

        # 批量处理
        results = {}
        for i, file_path in enumerate(files, 1):
            print(f"\n进度: [{i}/{len(files)}]")
            try:
                doc_ids = self.process_file(str(file_path), issemantic)
                results[str(file_path)] = doc_ids
            except Exception as e:
                print(f"✗ 处理失败: {file_path}")
                print(f"  错误: {e}")
                results[str(file_path)] = []

        return results

    def _extract_text(self, chunk) -> str:
        """
        从chunk中提取文本内容

        参数:
            chunk: 可能是str或Document对象

        返回:
            文本内容
        """
        if isinstance(chunk, str):
            return chunk
        elif hasattr(chunk, 'page_content'):
            # LangChain Document对象
            return chunk.page_content
        elif hasattr(chunk, 'content'):
            return chunk.content
        else:
            return str(chunk)
