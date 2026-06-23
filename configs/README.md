# Use-case configuration

Each YAML file in this directory defines a **deployment profile** for the RAG platform: corpus paths, Chroma collection, retrieval settings, prompt module, evaluation golden set, and UI branding.

Swap the active config at startup to run a different domain without code changes.

## Profiles

| File | Status | Domain |
|------|--------|--------|
| `mindbridge.yaml` | planned | Mental-health clinical copilot (reference demo) |
| `legal.yaml` | planned | Legal knowledge assistant |
| `finance.yaml` | planned | Finance / compliance Q&A |
| `hr.yaml` | planned | HR policy and handbook search |
| `education.yaml` | planned | Curriculum and course-material RAG |

Until the config loader is implemented, the platform runs with MindBridge defaults in `chains/` and `data/`. New profiles should follow the schema documented in the root [`README.md`](../README.md).
