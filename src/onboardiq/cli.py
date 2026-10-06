"""``onboardiq`` command-line entry point."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from onboardiq.config import Settings
from onboardiq.render import answer_plain
from onboardiq.types import LEVELS, ROLES


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "docs", None):
        settings.docs_dir = Path(args.docs)
    if getattr(args, "index_dir", None):
        settings.index_dir = Path(args.index_dir)
    return settings


def cmd_index(args: argparse.Namespace) -> int:
    from onboardiq.index.store import build_or_load
    from onboardiq.providers import make_embedder

    s = _settings(args)
    store, rebuilt = build_or_load(s.docs_dir, s.index_dir, make_embedder(s), s.chunk_size, s.chunk_overlap,
                                   force=args.force)
    state = "built" if rebuilt else "up to date (reused cached index)"
    sources = sorted({c.source for c in store.chunks})
    print(f"Index {state}: {len(store.chunks)} chunks from {len(sources)} documents -> {s.index_dir}")
    print(f"Embedder: {store.embedder_name}")
    return 0


def _assistant(args: argparse.Namespace):
    from onboardiq.service import Assistant

    return Assistant(_settings(args))


def cmd_ask(args: argparse.Namespace) -> int:
    answer = _assistant(args).ask(args.question, args.role, args.level)
    if args.json:
        payload = {"answer": answer.text, "refused": answer.refused, "role": answer.role, "level": answer.level,
                   "citations": [asdict(c) for c in answer.citations]}
        print(json.dumps(payload, indent=2))
    else:
        print(answer_plain(answer))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from onboardiq.feedback.store import FeedbackStore

    assistant = _assistant(args)
    feedback = FeedbackStore(assistant.settings.feedback_db)
    history: list[tuple[str, str]] = []
    print(f"onboardiq chat ({args.role}, {args.level}). Empty line to quit; '+' / '-' rates the last answer.")
    last = None
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line in ("+", "-"):
            if last is None:
                print("nothing to rate yet: ask a question first")
            else:
                feedback.submit(last, "helpful" if line == "+" else "not_helpful")
                print("feedback saved")
            continue
        last = assistant.ask(line, args.role, args.level, history)
        history.append((line, last.text))
        print("\n" + answer_plain(last))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from onboardiq.eval.runner import evaluate, format_report, load_golden

    assistant = _assistant(args)
    golden = load_golden(args.golden)
    results = {}
    for use_filter, label in ((True, "with role/level filter"), (False, "whole corpus, no filter")):
        report = evaluate(assistant.retriever, golden, use_filter=use_filter)
        results["filtered" if use_filter else "unfiltered"] = report
        print(format_report(report, len(golden), f"Retrieval on {len(golden)} questions, {label}"))
        print()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    from onboardiq.feedback.store import FeedbackStore

    store = FeedbackStore(_settings(args).feedback_db)
    if args.action == "export":
        count = store.export_csv(args.out)
        print(f"exported {count} feedback rows to {args.out}")
    else:
        print(json.dumps(store.summary(), indent=2))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onboardiq", description=__doc__)
    parser.add_argument("--docs", help="document folder (overrides ONBOARDIQ_DOCS_DIR)")
    parser.add_argument("--index-dir", help="index folder (overrides ONBOARDIQ_INDEX_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="build (or reuse) the persisted index")
    p.add_argument("--force", action="store_true", help="rebuild even if the documents did not change")
    p.set_defaults(func=cmd_index)

    for name, func in (("ask", cmd_ask), ("chat", cmd_chat)):
        p = sub.add_parser(name, help="ask one question" if name == "ask" else "interactive chat")
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--json", action="store_true", help="print JSON including citations")
        p.add_argument("--role", choices=ROLES, default="data_analyst")
        p.add_argument("--level", choices=LEVELS, default="mid")
        p.set_defaults(func=func)

    p = sub.add_parser("eval", help="retrieval recall@k / MRR on the golden set")
    p.add_argument("--golden", default="eval/golden_set.jsonl")
    p.add_argument("--out", help="also write the metrics as JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("feedback", help="summarise or export collected feedback")
    p.add_argument("action", choices=("summary", "export"))
    p.add_argument("--out", default="feedback_export.csv")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("ui", help="launch the Streamlit app (needs the 'ui' extra)")
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
