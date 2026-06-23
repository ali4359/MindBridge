"""Platform configuration: load and validate per-use-case YAML profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

RetrievalMode = Literal["bm25", "vector", "hybrid", "multi_query", "hybrid_rerank"]


class DownloadSourceConfig(BaseModel):
    """Remote PDF source for corpus download scripts."""

    model_config = ConfigDict(extra="forbid")

    category: str
    filename: str
    title: str
    license: str
    url: Optional[str] = None
    urls: Optional[list[str]] = None
    compiled: bool = False


class SessionNotesConfig(BaseModel):
    """Synthetic session-note generation settings for demo corpora."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "data/session_notes"
    generation_prompt: str = ""
    scenarios: list[str] = Field(default_factory=list)
    count: int = 20
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.8
    max_tokens: int = 900
    pause_seconds: float = 0.5


class DataSourceConfig(BaseModel):
    """Corpus paths, vector-store location, and embedding model."""

    model_config = ConfigDict(extra="forbid")

    sources_dir: str = "data"
    chroma_path: str = "data/chromadb"
    collection: str = "mindbridge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_sample_sentence: str = (
        "Cognitive behavioral therapy helps patients identify unhelpful thought patterns."
    )
    user_agent: str = (
        "MindBridge/0.1 (+https://github.com/mindbridge; research RAG demo; contact: local-dev)"
    )
    download_sources: list[DownloadSourceConfig] = Field(default_factory=list)
    session_notes: SessionNotesConfig = Field(default_factory=SessionNotesConfig)


class ChunkingConfig(BaseModel):
    """Text-splitting parameters for document indexing."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = 800
    chunk_overlap: int = 100
    separators: list[str] = Field(
        default_factory=lambda: [
            "\n\n",
            "\n",
            r"(?<=[.!?])\s+",
            " ",
            "",
        ]
    )
    is_separator_regex: bool = True
    min_chunks: int = 500


class RetrievalConfig(BaseModel):
    """Hybrid search, multi-query expansion, and reranking settings."""

    model_config = ConfigDict(extra="forbid")

    mode: RetrievalMode = "hybrid_rerank"
    top_k: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_top_n: int = 5
    multi_query_variants: int = 3
    enable_multi_query: bool = False
    enable_rerank: bool = True


class EntityConfig(BaseModel):
    """Domain entity taxonomy: document types, folder mapping, metadata schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = "patient"
    profile_fields: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(
        default_factory=lambda: ["guideline", "session_note", "research"]
    )
    category_to_doc_type: dict[str, str] = Field(
        default_factory=lambda: {
            "guidelines": "guideline",
            "session_notes": "session_note",
            "research": "research",
            "workbooks": "research",
        }
    )
    metadata_fields: list[str] = Field(
        default_factory=lambda: ["source", "doc_type", "page_number"]
    )


class PromptsConfig(BaseModel):
    """Prompt templates and module reference for the RAG chain."""

    model_config = ConfigDict(extra="forbid")

    prompt_module: str = "chains.prompts"
    system_role: Optional[str] = None
    system_message: Optional[str] = None
    user_template: Optional[str] = None
    context_chunk_template: Optional[str] = None
    multi_query_template: Optional[str] = None


class OutputSchemaConfig(BaseModel):
    """Expected answer shape and citation validation rules."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(default_factory=list)
    require_citations: bool = True
    pdf_citation_pattern: str = r"\.pdf\b"
    page_citation_pattern: str = r"\bp\.?\s*\d+"
    empty_context_message: str = "(No relevant context retrieved.)"


class SafetyConfig(BaseModel):
    """Audience restrictions, disclaimers, and prohibited model behaviours."""

    model_config = ConfigDict(extra="forbid")

    audience: str = "licensed mental health professionals"
    disclaimer: str = (
        "This assistant is a clinical knowledge tool, not a clinician. "
        "It does not diagnose, prescribe treatment, provide crisis intervention, "
        "or replace professional clinical judgment."
    )
    prohibited_capabilities: list[str] = Field(
        default_factory=lambda: [
            "diagnose",
            "prescribe",
            "crisis_intervention",
            "replace_clinical_judgment",
        ]
    )
    flag_patterns: list[str] = Field(default_factory=list)


class RetrievalBaselineCase(BaseModel):
    """Expected-source retrieval precision check."""

    model_config = ConfigDict(extra="forbid")

    query: str
    expected_source: str


class EvaluationConfig(BaseModel):
    """Golden sets and smoke-test queries for RAGAS and integration checks."""

    model_config = ConfigDict(extra="forbid")

    golden_set: Optional[str] = None
    ragas_target_scores: dict[str, float] = Field(default_factory=dict)
    retrieval_precision_threshold: int = 8
    retrieval_case_count: int = 10
    retrieval_top_n_check: int = 3
    retrieval_baseline_cases: list[RetrievalBaselineCase] = Field(default_factory=list)
    keyword_test_queries: list[str] = Field(default_factory=list)
    hybrid_test_query: str = "DBT distress tolerance skills"
    similarity_test_query: str = "CBT for depression"
    sample_queries: list[str] = Field(
        default_factory=lambda: [
            (
                "What does the NICE guideline recommend as first-line pharmacological "
                "treatment for depression in adults?"
            ),
            "What DBT distress tolerance skills can I teach a client for crisis survival?",
            (
                "How can I use a thought record worksheet with a client who has "
                "generalized anxiety?"
            ),
        ]
    )


class LlmConfig(BaseModel):
    """Groq LLM parameters for generation and query expansion."""

    model_config = ConfigDict(extra="forbid")

    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024


class UseCaseConfig(BaseModel):
    """Root profile loaded from a single YAML file — all platform layers read from this."""

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    description: str = ""

    data: DataSourceConfig = Field(default_factory=DataSourceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    entities: EntityConfig = Field(default_factory=EntityConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    output_schema: OutputSchemaConfig = Field(default_factory=OutputSchemaConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)

    def resolve_path(self, value: str, *, base: Path) -> Path:
        """Resolve a config path relative to ``base`` (typically the repo root)."""
        path = Path(value)
        if path.is_absolute():
            return path
        return (base / path).resolve()

    def sources_dir(self, *, base: Path) -> Path:
        return self.resolve_path(self.data.sources_dir, base=base)

    def chroma_path(self, *, base: Path) -> Path:
        return self.resolve_path(self.data.chroma_path, base=base)

    def golden_set_path(self, *, base: Path) -> Optional[Path]:
        if self.evaluation.golden_set is None:
            return None
        return self.resolve_path(self.evaluation.golden_set, base=base)


def load_config(path: str) -> UseCaseConfig:
    """Read a YAML use-case profile and return a validated ``UseCaseConfig``."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping, got {type(raw).__name__}: {config_path}"
        )

    return UseCaseConfig.model_validate(raw)
