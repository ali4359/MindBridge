# MindBridge Platform

A **config-driven RAG platform** for domain-specific knowledge assistants. The same codebase — ingestion engine, retrieval pipeline, API layer, and evaluation suite — powers any knowledge domain by swapping a single YAML config file.

**MindBridge** (mental-health clinical copilot) is the first use case deployed on the platform. Legal, finance, HR, education, and other domains follow with zero code changes — only a new config, corpus, and prompts.

## Platform vs. use case

| Layer | Shared (platform) | Per use case (config) |
|-------|-------------------|------------------------|
| Ingestion | PDF load, chunk, embed, index | Source paths, metadata schema |
| Retrieval | BM25 + vector hybrid, FlashRank rerank | Collection name, filters, top-k |
| Chains | LCEL RAG pipeline, citation formatting | System prompt, user template, sample queries |
| API | FastAPI endpoints | App title, CORS, auth hooks |
| Evaluation | RAGAS metrics and eval scripts | Golden Q&A set, domain rubric |
| UI | Streamlit shell (optional) | Branding, disclaimer, example questions |

One YAML file selects the active profile at startup. The mental-health profile ships as the default demo.

## Architecture

```
                    ┌─────────────────┐
                    │  config/*.yaml  │  ← swap use case here
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    ▼                        ▼                        ▼
 ingestion/            retrieval/               chains/
 (load · chunk ·        (BM25 · vector ·         (prompts · LLM ·
  embed · index)          rerank · pack)            RAG chain)
    │                        │                        │
    └────────────────────────┼────────────────────────┘
                             ▼
                      API / Streamlit
                             │
                      evaluation/ (RAGAS)
```

## Repository layout

```
MindBridge/
├── configs/        # Use-case YAML profiles (mindbridge, legal, …)
├── ingestion/      # PDF load, chunk, embed, index
├── retrieval/      # vector search, FlashRank rerank, hybrid fusion
├── chains/         # LangChain prompts, LLM, RAG pipeline
├── evaluation/     # RAGAS metrics and eval scripts
├── app/            # Streamlit UI (config-aware)
├── data/           # Corpus files + ChromaDB (not committed)
├── tests/
└── requirements.txt
```

## First use case: MindBridge (mental health)

The bundled demo indexes a mixed clinical corpus — NICE/WHO guidelines, CBT/DBT workbooks, and synthetic session notes — and answers therapist-facing questions with grounded, cited responses. See [`data/DEMO_STORY.md`](data/DEMO_STORY.md) for corpus provenance.

Future profiles (`legal`, `finance`, `hr`, `education`) reuse the same pipeline; only config, data, and prompts change.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add GROQ_API_KEY

# Download demo corpus (MindBridge profile)
python -m ingestion.download_sources

# Build vector index
python -m ingestion.indexer

# Run tests
pytest
```

## Configuration (planned shape)

Each use case is a YAML profile under `configs/`. Example sketch for the mental-health deployment:

```yaml
name: mindbridge
display_name: MindBridge Clinical Copilot
description: Grounded Q&A for licensed mental health professionals

data:
  sources_dir: data/
  chroma_path: data/chroma
  collection: mindbridge_clinical

retrieval:
  top_k: 8
  rerank_top_n: 4

chains:
  prompt_module: chains.prompts
  system_role: clinical_assistant

evaluation:
  golden_set: evaluation/golden/mindbridge.jsonl
```

The loader will resolve paths, import prompt templates, and wire retrievers from this file. Until that loader lands, the repo runs with the MindBridge defaults baked into `chains/` and `data/`.

## Dependencies

`langchain`, `langchain-community`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, `flashrank`, `groq`, `streamlit`, `ragas`, `pymupdf`

## Secrets

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Never commit credentials.

## Agent / contributor context

See [`AGENTS.md`](AGENTS.md) for module boundaries, task routing, and working agreements for Cursor agents.

## Licensing

Demo corpora remain subject to their publishers' terms. Synthetic session notes are fictional demo data only. Retrieved text is not medical, legal, or professional advice.
