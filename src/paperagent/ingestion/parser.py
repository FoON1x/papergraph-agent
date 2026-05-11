"""
文档解析模块

将PDF文档解析为ParsedDocument对象（只做解析，不做知识抽取）
"""

from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paperagent.config import Settings, get_settings
from paperagent.schemas import ParsedChunk, ParsedDocument, ParsedSection


class DocumentParser:
    """把 PDF 解析成可追溯的结构化文档对象。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def parse_pdf(self, path: Path) -> ParsedDocument:
        """解析单个 PDF，并返回 ParsedDocument。

        这是导入链路的第一步，主要负责：
        1. 校验输入文件；
        2. 生成稳定的 paper_id；
        3. 加载 PDF 文本；
        4. 切成带页码和顺序信息的 Chunk。
        """
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")

        paper_id = self._paper_id(path)
        # 当前 MVP 先把整篇文档视作一个 Section；后续可以扩展成真正的章节识别。
        section = ParsedSection(title="Document", order=0)
        documents = self.load_pdf_documents(path)
        section.chunks = self._documents_to_chunks(documents, paper_id, section.title, str(path))

        return ParsedDocument(
            paper_id=paper_id,
            source_path=path,
            title=path.stem,
            sections=[section],
            metadata={"parser": "unstructured"},
        )

    def parse_many(self, input_path: Path) -> list[ParsedDocument]:
        """解析单个 PDF 或一个目录下的全部 PDF。

        这个函数主要是为了批量场景准备，当前 CLI 的主入口最终还是走 Pipeline。
        """
        paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
        return [self.parse_pdf(path) for path in paths]

    def load_pdf_documents(self, path: Path) -> list[Document]:
        """借助 LangChain loader 读取 PDF，并切成 LangChain Document 列表。"""
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
        except ImportError as exc:
            raise RuntimeError("Install langchain-community and unstructured[pdf] to parse PDFs.") from exc

        loader = UnstructuredPDFLoader(str(path), mode="elements")
        loaded_documents = loader.load()
        split_documents = self.splitter.split_documents(loaded_documents)
        # 只保留真正有内容的文本块，避免空 Chunk 污染后续抽取和写图。
        documents = [doc for doc in split_documents if doc.page_content.strip()]
        if not documents:
            raise ValueError(f"No text could be extracted from {path}")
        return documents

    def _documents_to_chunks(
        self,
        documents: list[Document],
        paper_id: str,
        section_title: str,
        source_path: str,
    ) -> list[ParsedChunk]:
        """把 LangChain Document 列表转成项目内部的 ParsedChunk 列表。"""
        chunks: list[ParsedChunk] = []
        for index, document in enumerate(documents):
            chunks.append(
                ParsedChunk(
                    chunk_id=f"{paper_id}:chunk:{index}",
                    text=document.page_content,
                    section_title=section_title,
                    page_number=document.metadata.get("page_number") or document.metadata.get("page"),
                    order=index,
                    metadata={**document.metadata, "source_path": source_path},
                )
            )
        return chunks

    @staticmethod
    def _paper_id(path: Path) -> str:
        """根据 PDF 文件内容生成稳定的论文 ID。"""
        # 论文主键基于文件内容哈希，而不是文件名；这样重命名文件不会影响身份。
        digest = sha256(path.read_bytes()).hexdigest()[:16]
        return f"paper:{digest}"
