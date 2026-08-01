import argparse
import json
import sys
from contextlib import chdir
from pathlib import Path

from . import commands
from .agent import CodexAgent
from .errors import SparError
from .process import run_git
from .runner import DEFAULT_AGENT_TIMEOUT_SECONDS, run_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spar-run",
        description="Run a deterministic SPAR lifecycle with bounded external-agent phases.",
    )
    parser.add_argument("session_name", metavar="SESSION", help="initialized SPAR session")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        metavar="PATH",
        help="target Git repository (default: current directory)",
    )
    parser.add_argument("--model", required=True, help="explicit Codex model")
    parser.add_argument("--effort", required=True, help="explicit Codex reasoning effort")
    parser.add_argument(
        "--workflow",
        choices=("phased", "sequential-mcts", "persistent-proposer", "hybrid-spine"),
        default="phased",
        help="external-agent context policy (default: phased)",
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=int,
        default=DEFAULT_AGENT_TIMEOUT_SECONDS,
        metavar="SECONDS",
    )
    args = parser.parse_args(argv)
    if args.agent_timeout_seconds < 1:
        parser.error("--agent-timeout-seconds must be positive")
    repository = args.repo.expanduser().resolve()
    if not repository.is_dir():
        parser.error(f"repository is not a directory: {repository}")
    code_checkout_head = run_git(
        Path(__file__).resolve().parent, ["rev-parse", "HEAD"], check=False
    ).stdout.strip()
    try:
        with chdir(repository):
            max_parallel = commands.status(args.session_name)["max_parallel"]
            if args.workflow == "sequential-mcts" and max_parallel != 1:
                raise SparError("sequential-mcts workflow requires max_parallel = 1")
            if args.workflow == "hybrid-spine" and max_parallel != 3:
                raise SparError("hybrid-spine workflow requires max_parallel = 3")
            agent = CodexAgent(
                model=args.model,
                effort=args.effort,
                timeout_seconds=args.agent_timeout_seconds,
                preserve_context=args.workflow == "sequential-mcts",
            )
            proposal_agent = (
                CodexAgent(
                    model=args.model,
                    effort=args.effort,
                    timeout_seconds=args.agent_timeout_seconds,
                    preserve_context=True,
                )
                if args.workflow in {"persistent-proposer", "hybrid-spine"}
                else None
            )
            spine_agent = (
                CodexAgent(
                    model=args.model,
                    effort=args.effort,
                    timeout_seconds=args.agent_timeout_seconds,
                    preserve_context=True,
                )
                if args.workflow == "hybrid-spine"
                else None
            )
            researcher_context = {
                "sequential-mcts": "same Codex thread",
                "persistent-proposer": (
                    "persistent proposal thread; fresh implementation and reflection threads"
                ),
                "hybrid-spine": (
                    "one persistent propose/implement/reflect spine thread; "
                    "one persistent explorer proposal thread; fresh explorer "
                    "implementation and reflection threads"
                ),
            }.get(args.workflow, "fresh Codex thread per phase")
            result = run_session(
                args.session_name,
                agent,
                proposal_agent=proposal_agent,
                spine_agent=spine_agent,
                agent_metadata={
                    "adapter": "codex",
                    "model": args.model,
                    "effort": args.effort,
                    "timeout_seconds": args.agent_timeout_seconds,
                    "workflow": args.workflow,
                    "researcher_context": researcher_context,
                    "implementation_read_isolation": "not restricted to candidate worktree",
                    "code_checkout_head": code_checkout_head or None,
                },
            )
    except SparError as exc:
        print(f"spar-run: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
