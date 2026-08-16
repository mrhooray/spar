# Learnings

## Skills + tools vs. code-driven orchestration

SPAR was first implemented through skills, tools, and delegated subagents, with the primary researcher responsible for coordinating the research process. It later moved to a code-driven harness that owns orchestration while agents focus on proposing, implementing, and reflecting on interventions.

Both approaches improved evaluation outcomes. However, the skills-and-tools version did not consistently preserve role boundaries or enforce the candidate lifecycle. The primary researcher sometimes passed subagents excessive or inaccurate context, and deviations from the intended process often happened silently. Refining instructions after each observed deviation made prompts progressively larger without reliably improving adherence across long sessions. Compacting those accumulated instructions made them shorter but could introduce new interpretations and deviations.

The code-driven version was chosen because orchestration rules can be enforced directly and agent context can be controlled precisely. In addition, the sampled runs showed no clear outcome advantage from the skills-and-tools approach to offset its less reliable orchestration.

## Persistent agent sessions

Despite receiving the necessary task context, starting a fresh external coding-agent session for every proposal, implementation, and reflection significantly increased wall-clock time. Each agent turn still had to reconstruct its working understanding through repeated inspection. Persistent sessions avoid much of this repeated work and increase the chance of KV cache reuse.

SPAR now uses one continuing researcher context for proposals and one fresh worker session per candidate, reused from implementation through reflection. As candidates live in separate worktrees, successive proposal turns need different working directories. Asking a resumed agent in the prompt to change directories proved unreliable, so SPAR starts each proposal turn in the selected parent's worktree. Some harnesses support resuming across directories directly, while others require mechanisms like forking.

## Parallelism: elapsed time vs. sample efficiency

Higher parallelism reduces elapsed time by processing candidates concurrently. However, concurrent proposals share the same starting evidence, so candidates may be admitted before earlier evaluation and reflection results can inform subsequent proposals. This can reduce sample efficiency, particularly early in a run, while completing candidates faster.

## Search strategy

SPAR began with UCT-based MCTS as its conceptual model but evolved into global UCB-guided tree search, ranking all expandable candidates by a combination of observed value and exploration. This is a practical simplification for an open-ended intervention space and modest candidate limits. Selection is isolated from the research loop, leaving room for other strategies, such as UCT with progressive widening.
