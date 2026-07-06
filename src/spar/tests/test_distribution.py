import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[3]
WORKFLOWS = ("spar-init", "spar-start")


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_canonical_skill_and_agent_alias(workflow: str) -> None:
    canonical = ROOT / "skills" / workflow / "SKILL.md"
    amp = ROOT / ".agents" / "skills" / workflow / "SKILL.md"

    assert canonical.is_file()
    assert not canonical.is_symlink()
    assert amp.is_symlink()
    assert amp.resolve() == canonical

    metadata = (canonical.parent / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert "allow_implicit_invocation: false" not in metadata


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_thin_command_aliases_load_canonical_skill(workflow: str) -> None:
    for alias in (
        ROOT / ".opencode" / "commands" / f"{workflow}.md",
        ROOT / ".pi" / "prompts" / f"{workflow}.md",
    ):
        content = alias.read_text(encoding="utf-8")
        assert f"`{workflow}` skill" in content
        assert "$ARGUMENTS" in content


def test_harness_manifests_reference_canonical_skills() -> None:
    codex = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    claude = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    pi = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

    assert codex["name"] == claude["name"] == pi["name"] == "spar"
    assert (ROOT / codex["skills"]).resolve() == ROOT / "skills"
    assert pi["type"] == "module"
    assert pi["main"] == pi["exports"] == "./.opencode/plugins/spar.ts"
    assert pi["keywords"] == ["opencode-plugin", "pi-package"]
    assert pi["pi"]["skills"] == ["./skills"]
    assert pi["pi"]["prompts"] == ["./.pi/prompts"]


def test_opencode_adapter_exports_bundled_project_root() -> None:
    plugin = (ROOT / ".opencode" / "plugins" / "spar.ts").read_text(encoding="utf-8")

    assert "realpathSync" in plugin
    assert "output.env.SPAR_ROOT = sparRoot" in plugin


def test_bundled_cli_runs_from_another_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "--frozen",
            "--no-dev",
            "spar-cli",
            "--help",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "candidate-start" in result.stdout


def test_spar_start_names_the_research_loop_commands() -> None:
    content = (ROOT / "skills" / "spar-start" / "SKILL.md").read_text(encoding="utf-8")

    assert "CLI help is authoritative" in content
    for command in (
        "status",
        "parents",
        "candidate-inspect",
        "candidate-start",
        "candidate-evaluate",
        "candidate-profile",
        "candidate-complete",
        "candidate-fail",
    ):
        assert f"`{command}`" in content


def test_legacy_command_source_is_removed() -> None:
    assert not (ROOT / "commands").exists()
    assert not (ROOT / ".claude").exists()
