"""Load PDF documents with LangChain PyMuPDFLoader and MindBridge metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DocType = Literal["guideline", "session_note", "research"]

# Parent folder under data/ -> doc_type tag for retrieval filters.
DOC_TYPE_BY_CATEGORY: dict[str, DocType] = {
    "guidelines": "guideline",
    "session_notes": "session_note",
    "research": "research",
    # CBT/DBT workbooks are educational reference material for the demo corpus.
    "workbooks": "research",
}

PDF_CATEGORIES: tuple[str, ...] = tuple(DOC_TYPE_BY_CATEGORY)


def infer_doc_type(path: Path, data_dir: Path = DATA_DIR) -> DocType:
    """Infer doc_type from a PDF's parent folder under data/."""
    try:
        category = path.resolve().relative_to(data_dir.resolve()).parts[0]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot infer doc_type for {path} (expected under {data_dir})") from exc

    doc_type = DOC_TYPE_BY_CATEGORY.get(category)
    if doc_type is None:
        raise ValueError(
            f"Unknown category {category!r} for {path.name}; "
            f"expected one of {sorted(DOC_TYPE_BY_CATEGORY)}"
        )
    return doc_type


def _tag_page_documents(
    pages: list[Document],
    *,
    source: str,
    doc_type: DocType,
) -> list[Document]:
    """Normalize per-page metadata to MindBridge fields."""
    tagged: list[Document] = []
    for doc in pages:
        page_index = doc.metadata.get("page", 0)
        if not isinstance(page_index, int):
            raise ValueError(f"Expected integer page index in metadata, got {page_index!r}")
        tagged.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source,
                    "doc_type": doc_type,
                    "page_number": page_index + 1,
                },
            )
        )
    return tagged


def load_pdf(
    pdf_path: Path | str,
    *,
    doc_type: DocType | None = None,
    data_dir: Path = DATA_DIR,
) -> list[Document]:
    """Load one PDF as one LangChain Document per page with MindBridge metadata."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    resolved_type = doc_type or infer_doc_type(path, data_dir=data_dir)
    loader = PyMuPDFLoader(str(path), mode="page")
    pages = loader.load()
    documents = _tag_page_documents(
        pages,
        source=path.name,
        doc_type=resolved_type,
    )
    logger.info("Loaded %s — %d page(s) [%s]", path.name, len(documents), resolved_type)
    return documents


def load_all_pdfs(
    data_dir: Path = DATA_DIR,
    categories: tuple[str, ...] = PDF_CATEGORIES,
) -> list[Document]:
    """Load every PDF under the configured data/ subfolders."""
    all_documents: list[Document] = []
    for category in categories:
        category_dir = data_dir / category
        if not category_dir.is_dir():
            logger.warning("Skipping missing directory %s", category_dir)
            continue

        pdf_paths = sorted(category_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.info("No PDFs in %s", category_dir)
            continue

        doc_type = DOC_TYPE_BY_CATEGORY[category]
        for pdf_path in pdf_paths:
            all_documents.extend(load_pdf(pdf_path, doc_type=doc_type, data_dir=data_dir))

    logger.info(
        "Loaded %d page document(s) from %d PDF(s)",
        len(all_documents),
        sum(1 for c in categories for _ in (data_dir / c).glob("*.pdf") if (data_dir / c).is_dir()),
    )
    return all_documents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    documents = load_all_pdfs()
    by_source: dict[str, int] = {}
    for doc in documents:
        by_source[doc.metadata["source"]] = by_source.get(doc.metadata["source"], 0) + 1
    for source, count in sorted(by_source.items()):
        logger.info("%s — %d page(s)", source, count)
    logger.info("Total: %d page document(s)", len(documents))


if __name__ == "__main__":
    main()
