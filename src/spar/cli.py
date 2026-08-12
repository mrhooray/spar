import argparse
import json
import sys
import time
from contextlib import chdir
from pathlib import Path
from typing import Any

from . import loop
from .error import SparError
from .operation import candidate as candidate_ops
from .operation import session as session_ops


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.time_ns() // 1_000_000
    started_monotonic = time.monotonic_ns()
    try:
        repository = args.repo.expanduser()
        if not repository.is_dir():
            raise SparError(f"repository is not a directory: {repository}")
        with chdir(repository):
            result = dispatch(args)
    except SparError as exc:
        print(f"spar: {exc}", file=sys.stderr)
        return 1

    if args.command == "init":
        print(f"Initialized session {result['session_name']}.")
        print()
        print("Edit:")
        print(f"  objective: {result['objective_path']}")
        print(f"  config:    {result['config_path']}")
        print()
        print(f"Then run:\n  spar start {result['session_name']}")
        return 0

    if args.command == "start":
        return 0 if result["session"]["status"] == "completed" else 1

    result["cli_timing"] = {
        "started_at": started_at,
        "completed_at": time.time_ns() // 1_000_000,
        "elapsed_ms": (time.monotonic_ns() - started_monotonic) // 1_000_000,
        "command": args.command,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.command != "start" or result["session"]["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spar")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        metavar="PATH",
        help="target Git repository (default: current directory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="scaffold a new research session")
    init_parser.add_argument("session_name", metavar="SESSION", help="new session name")

    status_parser = subparsers.add_parser("status", help="show session state")
    status_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")

    stop_parser = subparsers.add_parser("stop", help="request a graceful session stop")
    stop_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")

    top_parser = subparsers.add_parser("top", help="show top MCTS candidates")
    top_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    top_parser.add_argument(
        "--k",
        type=int,
        default=session_ops.DEFAULT_TOP_LIMIT,
        metavar="COUNT",
        help=f"number of candidates to return (default: {session_ops.DEFAULT_TOP_LIMIT})",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="inspect one candidate",
        description=("Show candidate state, lineage, recorded operations, and retained artifact paths."),
    )
    inspect_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    inspect_parser.add_argument("candidate_id", metavar="CANDIDATE", help="candidate ID")

    start_parser = subparsers.add_parser("start", help="start or continue a research session")
    start_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    match args.command:
        case "init":
            return session_ops.init(args.session_name)
        case "status":
            return session_ops.status(args.session_name)
        case "stop":
            return session_ops.request_stop(args.session_name)
        case "top":
            return session_ops.top(args.session_name, k=args.k)
        case "inspect":
            return candidate_ops.inspect(args.session_name, args.candidate_id)
        case "start":
            return loop.ResearchLoop(
                args.session_name,
                progress=lambda line: print(line, file=sys.stderr, flush=True),
            ).start()
    raise AssertionError(f"unhandled command: {args.command}")
