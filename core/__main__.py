"""Platform CLI — swap use cases via USE_CASE_CONFIG."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from core.chain import build_chain, response_appears_cited
from core.config import USE_CASE_CONFIG_ENV, load_active_config, resolve_config_path
from core.ingestion import index_corpus
from core.llm import build_llm
from core.output_parser import build_output_schema
from core.retriever import build_retriever
from core.safety import apply_safety, match_flag_patterns
from core.vectorstore import load_vectorstore

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_active_config(args.config)
    vectorstore = index_corpus(config, reset=not args.no_reset)
    print(
        f"Indexed {vectorstore._collection.count()} vectors "
        f"for {config.display_name} → {config.chroma_path(base=REPO_ROOT)}"
    )
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    config = load_active_config(args.config)
    vectorstore = load_vectorstore(config)
    retriever = build_retriever(config, vectorstore, llm=build_llm(config))
    chain = build_chain(config, retriever, build_llm(config))

    answer = chain.invoke(args.question)
    print(f"\n=== {config.display_name} ===")
    print(f"Q: {args.question}\n")
    print(f"A: {answer}\n")

    if config.output_schema.require_citations:
        cited = response_appears_cited(answer, config)
        print(f"Citations detected: {cited}")

    safety = apply_safety(answer, config)
    print(f"Safety pass: {safety['safe']}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    config = load_active_config(args.config)
    config_path = resolve_config_path(args.config)
    question = args.question or (
        config.evaluation.sample_queries[0]
        if config.evaluation.sample_queries
        else "Summarize the key points in the retrieved documents."
    )

    print(f"Config: {config_path}")
    print(f"Profile: {config.display_name} (entity={config.entities.name})")

    schema_model = build_output_schema(config)
    print(f"Output schema: {schema_model.__name__} {list(schema_model.model_fields)}")

    vectorstore = load_vectorstore(config)
    retriever = build_retriever(config, vectorstore, llm=build_llm(config))
    docs = retriever.invoke(question)
    print(f"\nRetrieved {len(docs)} chunk(s):")
    for index, doc in enumerate(docs[:3], start=1):
        print(
            f"  [{index}] {doc.metadata.get('source')} "
            f"(p.{doc.metadata.get('page_number')}, {doc.metadata.get('doc_type')})"
        )

    chain = build_chain(config, retriever, build_llm(config))
    answer = chain.invoke(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")

    unsafe_probe = "you will definitely win — guaranteed outcome"
    print(f"\nSafety probe: {unsafe_probe!r}")
    print(f"  matched patterns: {bool(match_flag_patterns(unsafe_probe, config))}")
    print(f"  fallback: {apply_safety(unsafe_probe, config)['text'][:120]}...")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="MindBridge platform CLI — select profile with USE_CASE_CONFIG",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML profile path (or set ${USE_CASE_CONFIG_ENV}); required if neither is set",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load PDFs, chunk, embed, and index")
    ingest.add_argument("--no-reset", action="store_true", help="Append without clearing index")
    ingest.set_defaults(func=_cmd_ingest)

    query = sub.add_parser("query", help="Run a single RAG query")
    query.add_argument("question", help="Question to ask the active profile")
    query.set_defaults(func=_cmd_query)

    demo = sub.add_parser("demo", help="Show retrieval, answer, schema, and safety for a profile")
    demo.add_argument("--question", default=None, help="Override the default sample query")
    demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
