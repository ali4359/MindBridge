# MindBridge demo data story

This folder holds the **demo corpus** for MindBridge: clinical guidelines, therapy workbooks, and synthetic session notes used to show retrieval-augmented Q&A across mixed mental-health sources.

## How the corpus was built

### 1. Clinical guidelines (PDF)

| File | Source |
|------|--------|
| `guidelines/nice-ng222-depression.pdf` | [NICE NG222](https://www.nice.org.uk/guidance/ng222) — depression in adults |
| `guidelines/nice-cg113-anxiety.pdf` | [NICE CG113](https://www.nice.org.uk/guidance/cg113) — GAD and panic disorder |
| `guidelines/who-mhgap-intervention-guide-v2.pdf` | [WHO mhGAP-IG v2.0](https://www.who.int/publications/i/item/9789241549790) |

Downloaded with `requests`, validated with **PyMuPDF** (page count + text extraction).

### 2. CBT / DBT workbooks (open access PDFs)

| File | Source |
|------|--------|
| `workbooks/think-cbt-workbook.pdf` | [Think CBT Workbook](https://www.thinkcbt.com/thinkcbt-workbook) |
| `workbooks/cbt-worksheets-getselfhelp.pdf` | [GetSelfHelp.co.uk](https://www.getselfhelp.co.uk/) CBT worksheets (merged) |
| `workbooks/dbt-skills-workbook-mckay.pdf` | DBT skills workbook (McKay et al., hosted sample PDF) |

These provide structured exercises and skills language that complements formal guidelines.

### 3. Synthetic therapy session notes (LLM-generated)

Twenty fictional SOAP-format session notes live in `session_notes/`. They are **not real patient records** — they were generated to simulate a clinic’s de-identified note archive for RAG evaluation.

**Prompt (abbreviated):**

> Write one realistic but entirely fictional therapy session note in SOAP format (250–400 words). Outpatient CBT/DBT-informed care. Include presenting concern, interventions, client response, risk screen, and homework. Scenario: *{numbered scenario}*.

The full prompt is defined in `ingestion/generate_session_notes.py` as `SESSION_NOTE_PROMPT`. Generation uses the **Groq API** (`llama-3.3-70b-versatile`) with `GROQ_API_KEY` from `.env`.

**Why synthetic notes in the demo?**

- Shows MindBridge retrieving across **guidelines + workbooks + operational clinical text**.
- Avoids using or leaking real PHI.
- Gives RAGAS-style eval questions a third document type (short, episodic notes vs. long PDFs).

## Reproduce locally

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Download and validate PDFs
python -m ingestion.download_sources

# Generate 20 synthetic session notes (requires GROQ_API_KEY)
python -m ingestion.generate_session_notes
```

## Corpus layout

```
data/
├── guidelines/          # NICE + WHO PDFs
├── workbooks/           # CBT/DBT PDFs
├── session_notes/       # synthetic .txt notes
└── DEMO_STORY.md        # this file
```

## Licensing reminder

Guidelines and workbooks remain subject to their publishers’ terms. The synthetic notes are demo-only fiction. Do not treat retrieved text as medical advice.
