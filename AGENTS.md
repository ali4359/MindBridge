# MindBridge — Agent Instructions

This file is the canonical project brief for Cursor agents. **Read it at the start of every chat.**

## What this project is

MindBridge is a Python RAG system: ingest documents, store embeddings in ChromaDB, retrieve and rerank context, answer questions via a Groq-backed LangChain chain, evaluate with RAGAS, and expose a Streamlit UI.

## Repository layout

```
MindBridge/
├── ingestion/      # PDF load, chunk, embed, index
├── retrieval/      # vector search, FlashRank rerank, context packing
├── chains/         # LangChain QA / pipeline logic
├── evaluation/     # RAGAS metrics and eval scripts
├── app/            # Streamlit application
├── data/           # source files + ChromaDB (not committed)
├── requirements.txt
└── .venv/
```

## Dependencies

`langchain`, `langchain-community`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, `flashrank`, `groq`, `streamlit`, `ragas`, `pymupdf`

## Working agreements

- **Module boundaries**: Each top-level folder owns one concern. The Streamlit app imports from other modules; lower layers must not import from `app/`.
- **Secrets**: Use `.env` for API keys. Never commit credentials.
- **Changes**: Small, focused diffs. Reuse LangChain types and patterns already in the codebase.
- **Running code**: `source .venv/bin/activate` then run from the repo root.

## Task routing

| If the task involves… | Work in… |
|----------------------|----------|
| Loading or indexing documents | `ingestion/` |
| Search, similarity, reranking | `retrieval/` |
| Prompts, LLM chains, RAG logic | `chains/` |
| Measuring answer quality | `evaluation/` |
| UI, user interaction | `app/` |
| Sample PDFs or vector DB files | `data/` |

## Cursor rules

Persistent agent context also lives in `.cursor/rules/` (always applied). Keep `AGENTS.md` and those rules aligned when architecture changes.
