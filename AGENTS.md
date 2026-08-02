# MindBridge Platform — Agent Instructions

This file is the canonical project brief for Cursor agents. **Read it at the start of every chat.**

## What this project is

A **config-driven RAG platform**: ingest documents, store embeddings in ChromaDB, retrieve and rerank context, answer questions via a Groq-backed LangChain chain, evaluate with RAGAS, and expose a FastAPI / Streamlit layer.

**Use case = configuration.** The same ingestion engine, retrieval pipeline, API, and evaluation suite power any knowledge domain by swapping a YAML config file. **MindBridge** (mental-health clinical copilot) is the first reference deployment; legal, finance, HR, and education profiles follow with zero code changes.

## Repository layout

```
MindBridge/
├── configs/        # Per-use-case YAML profiles
├── adapters/       # Vendor source-system adapters (Rauha, generic, …)
├── ingestion/      # PDF load, chunk, embed, index
├── retrieval/      # vector search, FlashRank rerank, hybrid fusion
├── chains/         # LangChain QA / pipeline logic (prompts swappable via config)
├── evaluation/     # RAGAS metrics and eval scripts
├── app/            # Streamlit / API entrypoints (config-aware)
├── backend/        # FastAPI platform API
├── data/           # source files + ChromaDB (not committed)
├── requirements.txt
└── .venv/
```

## Dependencies

`langchain`, `langchain-community`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, `flashrank`, `groq`, `streamlit`, `ragas`, `pymupdf`

## Working agreements

- **Platform first**: Shared pipeline code stays domain-agnostic. Domain-specific prompts, metadata labels, and sample queries belong in config or use-case modules — not hard-coded across `ingestion/` or `retrieval/`.
- **Module boundaries**: Each top-level folder owns one concern. `app/` imports from lower layers; lower layers must not import from `app/`.
- **Secrets**: Use `.env` for API keys. Never commit credentials.
- **Changes**: Small, focused diffs. Reuse LangChain types and patterns already in the codebase.
- **Running code**: `source .venv/bin/activate` then run from the repo root.

## Task routing

| If the task involves… | Work in… |
|----------------------|----------|
| Use-case YAML, profile switching | `configs/` |
| External EHR / source-system adapters | `adapters/` |
| Loading or indexing documents | `ingestion/` |
| Search, similarity, reranking | `retrieval/` |
| Prompts, LLM chains, RAG logic | `chains/` |
| Measuring answer quality | `evaluation/` |
| UI, API, user interaction | `app/` / `backend/` |
| Sample PDFs or vector DB files | `data/` |

## Reference use case

The **MindBridge** mental-health profile is the default demo: clinical guidelines, CBT/DBT workbooks, synthetic session notes. Corpus details live in `data/DEMO_STORY.md`. When adding platform features, preserve this profile as a working end-to-end example.

## Cursor rules

Persistent agent context also lives in `.cursor/rules/` (always applied). Keep `AGENTS.md`, `README.md`, and those rules aligned when architecture changes.
