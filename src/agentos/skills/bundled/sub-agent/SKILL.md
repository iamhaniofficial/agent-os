---
name: sub-agent
description: 'Route coding work to an external coding harness (Claude Code, Codex, Gemini CLI, Cursor, GitHub Copilot, OpenCode, Pi, Droid, Qwen, Kimi, Kiro, ...) as an ACP child session via sessions_spawn(runtime="acp"). Use when: the user names a coding agent ("have Codex fix this", "ask Claude Code to review"), wants a feature built, a PR reviewed, a refactor or a multi-file change done by a separate worker, or a meta-skill step needs a full coding agent. Read this before spawning, steering, waiting for, or cancelling a harness. NOT for: one-line edits (edit directly), reading code (use read tools), or delegating to another AgentOS agent (plain sessions_spawn without runtime).'
provenance:
  origin: openclaw-derived
  license: MIT
  upstream_url: https://github.com/openclaw/openclaw
  maintained_by: AgentOS
metadata:
  {
    "openclaw":
      {
        "emoji": "🧩",
        "requires":
          {
            "anyBins":
              [
                "claude",
                "codex",
                "gemini",
                "cursor-agent",
                "copilot",
                "opencode",
                "pi",
                "droid",
                "qwen",
                "kimi",
                "npx",
              ],
          },
      },
  }
---

# Sub-Agent: route coding work to an ACP harness

Derived from OpenClaw's `acp-router` skill (MIT). AgentOS drives external
coding agents ("harnesses") over the Agent Client Protocol. A harness runs as
a child session with its own transcript: you get structured progress, its
permission prompts go to the user's approval queue, and its final answer is
pushed back to you with a `Stats:` line (runtime, tokens, cost when known).

Do not scrape a harness through `exec_command` / `background_process` when
the ACP path works. That path is only the fallback at the end of this page.

## Pick the harness

Map what the user says to `acp_agent`:

| User says | `acp_agent` |
| --- | --- |
| "claude", "claude code" | `claude` |
| "codex", "openai codex" | `codex` |
| "gemini", "gemini cli" | `gemini` |
| "cursor", "cursor cli", "cursor agent" | `cursor` |
| "copilot", "github copilot" | `copilot` |
| "opencode" | `opencode` |
| "pi" | `pi` |
| "droid", "factory droid" | `droid` |
| "qwen", "qwen code" | `qwen` |
| "kimi", "kimi cli" | `kimi` |
| "kiro", "kiro cli" | `kiro` |
| "grok build" | `grok-build` (a harness, not the xAI model id) |
| "kilo", "kilocode" | `kilocode` |
| "iflow", "junie", "qoder", "trae", "mux", "devin", "fx", "pool", "mcode", "zeroclaw", "fast-agent", "antigravity", "openclaw" | the same word |
| nothing specific | omit `acp_agent` (the configured `[acp] default_agent`, usually `claude`) |

If the user asks for a specific harness, use that one. If it fails, report
the error and ask; do not silently switch harness or do the work yourself.

## Spawn

```json
{
  "task": "Fix the failing test in tests/test_api.py. Run pytest -q before you finish and report what changed.",
  "runtime": "acp",
  "acp_agent": "codex",
  "cwd": "/path/to/app",
  "mode": "oneshot"
}
```

- `task`: self-contained. The harness cannot see this conversation: include
  the goal, the files, the constraints, how to verify, and the exact output
  you need back.
- `cwd`: the project directory. Never the AgentOS state directory
  (`~/.agentos/state`) or the live AgentOS workspace.
- `mode`: `oneshot` (default) for a single job; the harness exits when it
  answers. `persistent` when you expect follow-ups in the same harness
  context (iterating on a feature, a conversation the user keeps steering).
- `harness_mode`: optional session mode the harness offers, for example
  `read-only` (Codex) or `plan` / `acceptEdits` (Claude Code). Use a
  read-only mode for reviews and investigations.
- `worktree: true` (with optional `worktree_name`, `worktree_base_ref`):
  run in a managed git worktree instead of the checkout. Use it for parallel
  workers on one repo and for anything you want isolated from the user's
  working tree.
- `resume_session_id`: continue an earlier harness session you or your
  children started.

## Wait

After spawning one or more harnesses, call `sessions_yield` with no
`session_key` and end your turn. Completions are pushed back to this session
as messages; do not poll in a loop or sleep. Tell the user once what is
running and where, then report again only when something changes (a worker
finished, failed, or needs an answer).

## Steer, cancel, close

- Follow-up for a `persistent` child: `sessions_send` to its session key
  (`agent:<you>:acp:<id>`). `followup` queues the text after the current
  turn; `steer` cancels the running turn and sends the new text instead.
- List children: `subagents` (ACP children carry `runtime: "acp"`).
- Stop one: `subagents` with `action="kill"` (cancel + close).
- Harness permission prompts reach the user through the normal approvals
  surface. If a child is waiting on one, say so; do not try to answer it.

## Safety rules

1. Never spawn a worker with permissions bypassed (`approve-all`, Claude
   `bypassPermissions`, Codex `danger-full-access`) on code you have not
   read: a pull request from a fork, a freshly cloned unknown repo, a branch
   someone else pushed. Hostile repo content can instruct the harness to run
   anything. Review those in a read-only `harness_mode`, in a worktree.
2. Never start a harness inside the AgentOS state or workspace directories,
   and never check out branches there; use the project checkout or a
   worktree.
3. One writer per checkout. Parallel workers on the same repo each get
   `worktree: true`.
4. Be patient. A quiet harness is working; only cancel on the user's word,
   a clear hang (no progress for a long time on a short task), or a wrong
   direction.

## When spawning fails

| Error code | What to do |
| --- | --- |
| `ACP_HARNESS_MISSING` | The harness CLI is not installed. Tell the user which one (`agentos acp doctor <id>` shows the fix). |
| `ACP_AGENT_NOT_ALLOWED` | Not in `[acp] allowed_agents`, or ACP is disabled. Report it and ask which allowed harness to use. |
| `ACP_SESSION_INIT_FAILED` | The harness started but would not open a session (often not logged in). Ask the user to run the harness CLI once and sign in. |
| `ACP_PERMISSION_PROMPT_UNAVAILABLE` | It needed approval and nobody can answer here. Report it; suggest a read-only task or a different `[acp] permission_mode`. |
| `ACP_TURN_TIMEOUT` | The turn hit `[acp] turn_timeout_s`. Report partial results; respawn with a smaller task if the user wants. |
| unknown parameter `runtime` / `ACP_UNAVAILABLE` | This build or host has no ACP runtime. Use the fallback below. |

## Fallback: raw CLI (only without the ACP runtime)

Use this only when `sessions_spawn(runtime="acp")` is unavailable on this
host (no ACP runtime, no Node/`npx` for the adapter). You lose approvals,
resume and cost accounting. Run the harness non-interactively in the project
directory with `background_process`, then follow it with `process`
(`poll`, `log`, `kill`):

```text
background_process(workdir="/path/to/project",
  command="codex exec --sandbox workspace-write 'Fix the failing test in tests/test_api.py and run pytest -q'")

background_process(workdir="/path/to/project",
  command="claude -p --permission-mode acceptEdits 'Fix the failing test in tests/test_api.py and run pytest -q'")

process(action="log", session_id="<id>")
```

- Codex: `codex exec` runs one task and exits. `--sandbox read-only` for
  reviews, `workspace-write` to let it edit; the model comes from
  `~/.codex/config.toml` (or `-m`). It refuses to run outside a git
  repository unless you add `--skip-git-repo-check`.
- Claude Code: `claude -p` (print mode) runs one task and exits.
  `--permission-mode plan` for read-only analysis, `acceptEdits` to let it
  edit files; tools it is not allowed to use are refused, not prompted.
  Do not add `--bare`: it skips the user's login.
- Never use `--dangerously-bypass-approvals-and-sandbox` (Codex) or
  `--dangerously-skip-permissions` / `bypassPermissions` (Claude) outside an
  externally sandboxed, trusted checkout (see safety rule 1).
