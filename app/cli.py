"""Command line interface.

python -m app.cli ingest              # offline: chunk + embed + index
python -m app.cli predict IMG [--model lfm3-vl-2b]
python -m app.cli evaluate [--model lfm3-vl-2b] [--limit N] [--no-resume]
python -m app.cli serve [--host H] [--port P]

Every run is paired: First Pass, description-driven retrieval, then a
RAG-assisted Second Pass with the same selected model.

`make eval` is the only command that writes the consolidated summary and the
comparison table; the CLI evaluates one model into results/<model_id>/.
"""

from __future__ import annotations

import argparse
import json
import sys

from app import __version__
from app.config import MODEL_REGISTRY


def _cmd_ingest(args: argparse.Namespace) -> int:
    from app.rag.ingestion import build_index

    stats = build_index()
    print(json.dumps(stats, indent=2))
    print(f"Index written to {stats['index_dir']}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from app.pipeline.orchestrator import predict

    response = predict(args.image, model_id=args.model)
    print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0 if response.status == "ok" else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from app.config import load_config
    from app.evaluation.runner import run_evaluation

    cfg = load_config()
    if args.model:
        cfg = cfg.for_model(args.model)
    results = cfg.results.resolve_dir()
    if args.dogs is not None:
        results /= "fast"  # sampled runs never mix with the full 200-image results
    model_dir = results / cfg.vision.model_id.replace("/", "_")
    summary = run_evaluation(
        limit=args.limit,
        resume=not args.no_resume,
        cfg=cfg,
        output_dir=model_dir,
        dogs=args.dogs,
    )
    print(json.dumps(summary, indent=2))
    print(f"results written to {model_dir}")
    print("run `make eval` for the consolidated summary and comparison table")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.config import get_config

    server = get_config().server
    uvicorn.run(
        "app.main:app", host=args.host or server.host, port=args.port or server.port, reload=False
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doggo-rag", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="build the knowledge index (offline)")
    ingest.set_defaults(func=_cmd_ingest)

    predict = sub.add_parser("predict", help="run the paired two-pass pipeline on one image")
    predict.add_argument("image")
    predict.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    predict.set_defaults(func=_cmd_predict)

    evaluate = sub.add_parser("evaluate", help="run the paired First/Second Pass evaluation")
    evaluate.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument(
        "--dogs", type=int, default=None, help="evaluate only N sampled dogs, plus all non-dogs"
    )
    evaluate.add_argument("--no-resume", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    serve = sub.add_parser("serve", help="start the FastAPI server")
    serve.add_argument("--host", default=None, help="default: server.host in the config")
    serve.add_argument("--port", type=int, default=None, help="default: server.port in the config")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
