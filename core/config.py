"""Platform configuration: load and validate per-use-case YAML profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RetrievalMode = Literal["bm25", "vector", "hybrid", "multi_query", "hybrid_rerank"]


class DataSourceEntryConfig(BaseModel):
    """Local corpus folder and document-type tag for ingestion."""

    model_config = ConfigDict(extra="forbid")

    folder: str
    doc_type: str


def _default_data_sources() -> list[DataSourceEntryConfig]:
    return [
        DataSourceEntryConfig(folder="guidelines", doc_type="guideline"),
        DataSourceEntryConfig(folder="session_notes", doc_type="session_note"),
        DataSourceEntryConfig(folder="research", doc_type="research"),
        DataSourceEntryConfig(folder="workbooks", doc_type="research"),
    ]


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
        "This sentence is used to verify the embedding model loads correctly."
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


class GraphNodeConfig(BaseModel):
    """Extractable graph node type declared in a use-case profile."""

    model_config = ConfigDict(extra="forbid")

    label: str
    extraction_key: str
    identity: Literal["id", "name"] = "name"


class GraphRelationshipConfig(BaseModel):
    """Directed edge between two node labels from the active graph schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: str
    from_: str = Field(alias="from")
    to: str


class GraphSchemaConfig(BaseModel):
    """Domain knowledge-graph vocabulary — labels and edges come from YAML only."""

    model_config = ConfigDict(extra="forbid")

    entity_node: str
    session_node: str
    entity_session_relationship: Optional[str] = None
    nodes: list[GraphNodeConfig] = Field(default_factory=list)
    relationships: list[GraphRelationshipConfig] = Field(default_factory=list)
    similarity_primary_node: Optional[str] = None
    similarity_secondary_node: Optional[str] = None

    @model_validator(mode="after")
    def _validate_graph_schema(self) -> GraphSchemaConfig:
        known_labels = {self.entity_node, self.session_node} | {
            node.label for node in self.nodes
        }

        for field_name, label in (
            ("similarity_primary_node", self.similarity_primary_node),
            ("similarity_secondary_node", self.similarity_secondary_node),
        ):
            if label is not None and label not in known_labels:
                raise ValueError(
                    f"graph_schema.{field_name}: unknown label {label!r}"
                )

        for rel in self.relationships:
            if rel.from_ not in known_labels:
                raise ValueError(
                    f"graph_schema.relationships: unknown from label {rel.from_!r}"
                )
            if rel.to not in known_labels:
                raise ValueError(
                    f"graph_schema.relationships: unknown to label {rel.to!r}"
                )

        extraction_keys = [node.extraction_key for node in self.nodes]
        if len(extraction_keys) != len(set(extraction_keys)):
            raise ValueError("graph_schema.nodes: duplicate extraction_key values")

        session_rels = [
            rel
            for rel in self.relationships
            if rel.from_ == self.entity_node and rel.to == self.session_node
        ]
        if self.entity_session_relationship is None:
            if len(session_rels) != 1:
                raise ValueError(
                    "graph_schema must define exactly one relationship from "
                    f"{self.entity_node!r} to {self.session_node!r}, or set "
                    "entity_session_relationship explicitly"
                )
            object.__setattr__(
                self, "entity_session_relationship", session_rels[0].type
            )
        elif not any(
            rel.type == self.entity_session_relationship for rel in session_rels
        ):
            raise ValueError(
                "graph_schema.entity_session_relationship "
                f"{self.entity_session_relationship!r} must match a relationship "
                f"from {self.entity_node!r} to {self.session_node!r}"
            )

        return self

    def known_labels(self) -> frozenset[str]:
        """All node labels referenced by this schema."""
        return frozenset(
            {self.entity_node, self.session_node, *(node.label for node in self.nodes)}
        )

    def node_by_label(self) -> dict[str, GraphNodeConfig]:
        """Map extractable node labels to their config entries."""
        return {node.label: node for node in self.nodes}

    def extraction_keys(self) -> dict[str, str]:
        """Map extraction JSON keys to Neo4j node labels."""
        return {node.extraction_key: node.label for node in self.nodes}


class EntityConfig(BaseModel):
    """Domain entity taxonomy: document types, folder mapping, metadata schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = "entity"
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
    """Prompt templates for the config-driven RAG chain."""

    model_config = ConfigDict(extra="forbid")

    system: str = ""
    entity_context_header: str = ""
    knowledge_header: str = ""
    question_header: str = ""
    response_requirements: str = ""
    empty_entity_context: str = "(No entity profile provided.)"
    context_chunk_template: str = ""
    multi_query_template: Optional[str] = None


class OutputSchemaConfig(BaseModel):
    """Expected answer shape and citation validation rules."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = ""
    fields: list[str] = Field(default_factory=list)
    require_citations: bool = True
    pdf_citation_pattern: str = r"\.pdf\b"
    page_citation_pattern: str = r"\bp\.?\s*\d+"
    empty_context_message: str = "(No relevant context retrieved.)"


class SafetyConfig(BaseModel):
    """Audience restrictions, disclaimers, and prohibited model behaviours."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    audience: str = "licensed professionals"
    disclaimer: str = (
        "This assistant is an informational tool, not a licensed professional. "
        "It does not replace professional judgment."
    )
    prohibited_capabilities: list[str] = Field(default_factory=list)
    flag_patterns: list[str] = Field(default_factory=list)
    fallback_message: str = (
        "I cannot provide that response because it may violate safety guidelines "
        "for this use case. Please rephrase your request or consult a qualified professional."
    )


class RetrievalBaselineCase(BaseModel):
    """Expected-source retrieval precision check."""

    model_config = ConfigDict(extra="forbid")

    query: str
    expected_source: str


class EvaluationConfig(BaseModel):
    """Golden sets and smoke-test queries for RAGAS and integration checks."""

    model_config = ConfigDict(extra="forbid")

    golden_set: Optional[str] = None
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    ragas_target_scores: dict[str, float] = Field(default_factory=dict)
    retrieval_precision_threshold: int = 8
    retrieval_case_count: int = 10
    retrieval_top_n_check: int = 3
    sample_pause_seconds: float = 0.0
    ragas_batch_size: Optional[int] = None
    retrieval_baseline_cases: list[RetrievalBaselineCase] = Field(default_factory=list)
    keyword_test_queries: list[str] = Field(default_factory=list)
    hybrid_test_query: str = ""
    similarity_test_query: str = ""
    sample_queries: list[str] = Field(default_factory=list)

    @property
    def target_scores(self) -> dict[str, float]:
        """Alias for ``ragas_target_scores`` used by the RAGAS runner."""
        return self.ragas_target_scores


class LlmConfig(BaseModel):
    """Groq LLM parameters for generation and query expansion."""

    model_config = ConfigDict(extra="forbid")

    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024


class RouterConfig(BaseModel):
    """Keyword signals used to route a query between RAG and agent handling."""

    model_config = ConfigDict(extra="forbid")

    rag_signals: list[str] = Field(default_factory=list)
    agent_signals: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """System prompt and settings for the agent-mode handler."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = ""


class SourceSystemConfig(BaseModel):
    """External system labels for the entity/session ingestion adapter."""

    model_config = ConfigDict(extra="forbid")

    entity_label: str = ""
    session_label: str = ""
    adapter: str = ""


class UseCaseConfig(BaseModel):
    """Root profile loaded from a single YAML file — all platform layers read from this."""

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    description: str = ""

    data_sources: list[DataSourceEntryConfig] = Field(default_factory=_default_data_sources)
    data: DataSourceConfig = Field(default_factory=DataSourceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    entities: EntityConfig = Field(default_factory=EntityConfig)
    graph_schema: Optional[GraphSchemaConfig] = None
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    output_schema: OutputSchemaConfig = Field(default_factory=OutputSchemaConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    router: Optional[RouterConfig] = None
    agent: Optional[AgentConfig] = None
    source_system: Optional[SourceSystemConfig] = None

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

    @property
    def langsmith_project(self) -> str:
        """LangSmith project name for isolated trace dashboards per profile."""
        return f"{self.name}-traces"


def _finalize_config(config: UseCaseConfig) -> UseCaseConfig:
    """Apply runtime side effects after a profile is validated."""
    from core.tracing import configure_langsmith_project

    configure_langsmith_project(config)
    return config


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

    return _finalize_config(UseCaseConfig.model_validate(raw))


USE_CASE_CONFIG_ENV = "USE_CASE_CONFIG"


def resolve_config_path(path: Optional[str] = None) -> str:
    """Resolve the active use-case YAML path from arg or ``USE_CASE_CONFIG``; no domain default."""
    if path:
        return path
    import os

    env_path = os.environ.get(USE_CASE_CONFIG_ENV)
    if not env_path:
        raise RuntimeError(
            f"No use-case profile selected: pass --config or set {USE_CASE_CONFIG_ENV} "
            "to a YAML profile path (see configs/)"
        )
    return env_path


def load_active_config(path: Optional[str] = None) -> UseCaseConfig:
    """Load the use-case profile selected by ``path`` or ``USE_CASE_CONFIG``."""
    return load_config(resolve_config_path(path))
