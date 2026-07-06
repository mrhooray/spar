---
name: spar-init
description: Initialize a SPAR research session
---

# Initialize Session

1. Collect the target Git repository and session name, showing available defaults and asking when either is missing or ambiguous. Default to the repository containing the current working directory. The session name must be one path segment.

2. Scaffold the session with `spar-cli`. `SPAR_ROOT` identifies the SPAR source directory. Use its existing value, or derive and set it based on this skill's directory. Run the CLI with:

```sh
uv run --project "$SPAR_ROOT" --frozen --no-dev spar-cli \
  --repo <target-repository> init <session-name>
```

   This creates `<target-repository>/.spar/<session-name>/`, records the repository's current commit as the baseline, and adds `.spar/` to `.git/info/exclude`.

   If the user requested scaffolding only, report the returned `objective_path` and `config_path`, tell them to complete both files, and stop here.

3. Read and show the generated `objective_path` and `config_path` templates. Collect the user's objective and required configuration. Ask whether they want to customize any optional settings shown in `config.toml`.
4. Present the proposed objective and configuration for review. Revise them until the user confirms them.
5. Write the confirmed content to `objective.md` and `config.toml`, preserving unspecified configuration defaults.
6. Report the session name and both setup paths, and say the session is ready for `spar-start`. Stop without running evaluation or research.

Use only user-supplied or user-confirmed setup. Do not invent or broaden the objective, and do not infer a canonical evaluation command without confirmation.
