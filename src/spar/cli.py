from contextlib import chdir
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

from . import commands
from .errors import SparError
from .state import DECISION_DISCARD, DECISION_KEEP


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
        print(f"spar-cli: {exc}", file=sys.stderr)
        return 1

    result["cli_timing"] = {
        "started_at": started_at,
        "completed_at": time.time_ns() // 1_000_000,
        "elapsed_ms": (time.monotonic_ns() - started_monotonic) // 1_000_000,
        "command": args.command,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spar-cli")
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

    parents_parser = subparsers.add_parser("parents", help="show top MCTS parent candidates for expansion")
    parents_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    parents_parser.add_argument(
        "--k",
        type=int,
        default=commands.DEFAULT_PARENT_LIMIT,
        metavar="COUNT",
        help=f"number of parent suggestions to return (default: {commands.DEFAULT_PARENT_LIMIT})",
    )

    inspect_parser = subparsers.add_parser("candidate-inspect", help="inspect one candidate and its artifacts")
    inspect_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    inspect_parser.add_argument("candidate_id", metavar="CANDIDATE", help="candidate ID")

    start_parser = subparsers.add_parser("candidate-start", help="create a candidate and isolated worktree")
    start_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    start_parser.add_argument("--parent", required=True, metavar="CANDIDATE", help="completed, kept parent ID")
    start_parser.add_argument("--hypothesis", required=True, metavar="TEXT", help="testable expected outcome")
    start_parser.add_argument("--instructions", required=True, metavar="TEXT", help="one intervention to implement")
    start_parser.add_argument(
        "--rationale", required=True, metavar="TEXT", help="reason to test this intervention from the selected parent"
    )

    evaluate_parser = subparsers.add_parser("candidate-evaluate", help="run the configured evaluation command")
    evaluate_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    evaluate_parser.add_argument("candidate_id", metavar="CANDIDATE", help="implementing or evaluating candidate ID")

    profile_parser = subparsers.add_parser("candidate-profile", help="run the configured profiling command")
    profile_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    profile_parser.add_argument("candidate_id", metavar="CANDIDATE", help="reflecting or completed candidate ID")

    complete_parser = subparsers.add_parser("candidate-complete", help="record a decision and update MCTS")
    complete_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    complete_parser.add_argument("candidate_id", metavar="CANDIDATE", help="reflecting or finalizing candidate ID")
    complete_parser.add_argument("--summary", required=True, metavar="TEXT", help="result and key learnings")
    complete_parser.add_argument(
        "--decision",
        choices=(DECISION_KEEP, DECISION_DISCARD),
        required=True,
        help="whether the candidate remains eligible as a future parent",
    )
    complete_parser.add_argument("--decision-reason", required=True, metavar="TEXT", help="reason for the decision")

    fail_parser = subparsers.add_parser("candidate-fail", help="close a candidate without a trustworthy result")
    fail_parser.add_argument("session_name", metavar="SESSION", help="initialized session name")
    fail_parser.add_argument("candidate_id", metavar="CANDIDATE", help="unfinished candidate ID")
    fail_parser.add_argument("--error", required=True, metavar="TEXT", help="failure summary")
    fail_parser.add_argument("--interrupted", action="store_true", help="record the candidate as interrupted")

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    match args.command:
        case "init":
            return commands.init(args.session_name)
        case "status":
            return commands.status(args.session_name)
        case "parents":
            return commands.parents(args.session_name, k=args.k)
        case "candidate-inspect":
            return commands.candidate_inspect(args.session_name, args.candidate_id)
        case "candidate-start":
            return commands.candidate_start(
                args.session_name,
                parent_id=args.parent,
                hypothesis=args.hypothesis,
                instructions=args.instructions,
                rationale=args.rationale,
            )
        case "candidate-evaluate":
            return commands.candidate_evaluate(args.session_name, args.candidate_id)
        case "candidate-profile":
            return commands.candidate_profile(args.session_name, args.candidate_id)
        case "candidate-complete":
            return commands.candidate_complete(
                args.session_name,
                args.candidate_id,
                summary=args.summary,
                decision=args.decision,
                decision_reason=args.decision_reason,
            )
        case "candidate-fail":
            return commands.candidate_fail(
                args.session_name,
                args.candidate_id,
                args.error,
                interrupted=args.interrupted,
            )
    raise AssertionError(f"unhandled command: {args.command}")
