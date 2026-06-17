"""Download and validate MindBridge demo data sources (PDFs)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import requests

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

USER_AGENT = (
    "MindBridge/0.1 (+https://github.com/mindbridge; research RAG demo; "
    "contact: local-dev)"
)

# Free, publicly hosted PDFs for the MindBridge demo corpus.
SOURCES: tuple[dict[str, str], ...] = (
    {
        "category": "guidelines",
        "filename": "nice-ng222-depression.pdf",
        "title": "NICE NG222 — Depression in adults: treatment and management",
        "url": (
            "https://www.nice.org.uk/guidance/ng222/resources/"
            "depression-in-adults-treatment-and-management-pdf-66143832307909"
        ),
        "license": "© NICE — free for personal/educational use via nice.org.uk",
    },
    {
        "category": "guidelines",
        "filename": "nice-cg113-anxiety.pdf",
        "title": "NICE CG113 — GAD and panic disorder in adults: management",
        "url": (
            "https://www.nice.org.uk/guidance/cg113/resources/"
            "generalised-anxiety-disorder-and-panic-disorder-in-adults-management-pdf-35109387756997"
        ),
        "license": "© NICE — free for personal/educational use via nice.org.uk",
    },
    {
        "category": "guidelines",
        "filename": "who-mhgap-intervention-guide-v2.pdf",
        "title": "WHO mhGAP Intervention Guide v2.0",
        # IRIS HTML shell blocks naive bitstream URLs; DSpace API content link is stable.
        "url": (
            "https://iris.who.int/server/api/core/bitstreams/"
            "6ded7ffd-9d69-493a-b48a-0b3e6250c173/content"
        ),
        "license": "© WHO — non-commercial use; see who.int copyright terms",
    },
    {
        "category": "workbooks",
        "filename": "think-cbt-workbook.pdf",
        "title": "Think CBT Workbook (Psychology Tools / Think CBT)",
        "url": "https://www.thinkcbt.com/images/Downloads/PRINT-WORKBOOK-THINK-CBT-V-09.05.17.pdf",
        "license": "© Think CBT — free static PDF for personal/self-help use",
    },
    {
        "category": "workbooks",
        "filename": "cbt-worksheets-getselfhelp.pdf",
        "title": "CBT Worksheets compilation (GetSelfHelp.co.uk)",
        "urls": [
            "https://www.getselfhelp.co.uk/docs/ThoughtRecordSheet7.pdf",
            "https://www.getselfhelp.co.uk/docs/UnhelpfulThinkingHabits.pdf",
            "https://www.getselfhelp.co.uk/docs/ACELog.pdf",
            "https://www.getselfhelp.co.uk/docs/ActivityDiary.pdf",
            "https://www.get.gg/docs/ProblemSolvingWorksheet.pdf",
        ],
        "license": "© GetSelfHelp.co.uk — free CBT resources for personal use",
        "compiled": True,
    },
    {
        "category": "workbooks",
        "filename": "dbt-skills-workbook-mckay.pdf",
        "title": "The Dialectical Behavior Therapy Skills Workbook (McKay et al.)",
        "url": "https://morethantherapy.org/assets/files/The-DBTTherapySkillsWorkbook.pdf",
        "license": "© New Harbinger — sample/educational hosting; not for redistribution",
    },
)


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    pages: int
    chars: int
    sha256: str


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def merge_pdfs(urls: Iterable[str], dest: Path, session: requests.Session) -> bytes:
    """Download multiple PDFs and merge into one workbook with PyMuPDF."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged = fitz.open()
    try:
        for url in urls:
            response = session.get(url, timeout=120, allow_redirects=True)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"Expected PDF from {url}")
            source = fitz.open(stream=response.content, filetype="pdf")
            merged.insert_pdf(source)
            source.close()
        content = merged.tobytes()
        dest.write_bytes(content)
        return content
    finally:
        merged.close()


def download_pdf(url: str, dest: Path, session: requests.Session | None = None) -> bytes:
    """Download a PDF and return raw bytes."""
    sess = session or _session()
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = sess.get(url, timeout=120, allow_redirects=True)
    response.raise_for_status()

    content = response.content
    if not content.startswith(b"%PDF"):
        raise ValueError(f"Expected PDF from {url}, got {response.headers.get('content-type')}")

    dest.write_bytes(content)
    return content


def validate_pdf(path: Path) -> tuple[int, int]:
    """Open with PyMuPDF; return (page_count, total_extracted_chars)."""
    doc = fitz.open(path)
    try:
        pages = doc.page_count
        text = "".join(page.get_text() for page in doc)
        return pages, len(text.strip())
    finally:
        doc.close()


def prepare_source(
    source: dict[str, str],
    data_dir: Path = DATA_DIR,
    session: requests.Session | None = None,
    force: bool = False,
) -> DownloadResult:
    """Download (if needed) and validate a single source PDF."""
    dest = data_dir / source["category"] / source["filename"]
    if dest.exists() and not force:
        logger.info("Skipping existing %s", dest.name)
    else:
        logger.info("Downloading %s", source["title"])
        sess = session or _session()
        if source.get("compiled"):
            merge_pdfs(source["urls"], dest, session=sess)
        else:
            download_pdf(source["url"], dest, session=sess)

    pages, chars = validate_pdf(dest)
    if pages < 1 or chars < 500:
        raise ValueError(f"{dest.name} looks empty or corrupt ({pages} pages, {chars} chars)")

    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return DownloadResult(path=dest, pages=pages, chars=chars, sha256=digest)


def prepare_all(
    data_dir: Path = DATA_DIR,
    sources: Iterable[dict[str, str]] = SOURCES,
    force: bool = False,
) -> list[DownloadResult]:
    """Download and validate all configured sources."""
    session = _session()
    results: list[DownloadResult] = []
    for source in sources:
        results.append(prepare_source(source, data_dir=data_dir, session=session, force=force))
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = prepare_all()
    for result in results:
        logger.info(
            "%s — %d pages, %d chars, sha256=%s…",
            result.path.name,
            result.pages,
            result.chars,
            result.sha256[:12],
        )
    logger.info("Prepared %d PDFs under %s", len(results), DATA_DIR)


if __name__ == "__main__":
    main()
