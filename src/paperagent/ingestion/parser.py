from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paperagent.config import Settings, get_settings
from paperagent.schemas import ParsedChunk, ParsedDocument, ParsedSection


class DocumentParser:
    """Parse PDF files into source-traceable chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def parse_pdf(self, path: Path) -> ParsedDocument:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")

        paper_id = self._paper_id(path)
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
        paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
        return [self.parse_pdf(path) for path in paths]

    def load_pdf_documents(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
        except ImportError as exc:
            raise RuntimeError("Install langchain-community and unstructured[pdf] to parse PDFs.") from exc

        loader = UnstructuredPDFLoader(str(path), mode="elements")
        loaded_documents = loader.load()
        split_documents = self.splitter.split_documents(loaded_documents)
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
        digest = sha256(path.read_bytes()).hexdigest()[:16]
        return f"paper:{digest}"
