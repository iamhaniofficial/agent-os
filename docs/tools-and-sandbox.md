# Tools, Approvals, and Sandbox

AgentOS tools give the agent useful capabilities. Policy layers, approval
surfaces, workspace constraints, and sandbox posture control how those tools are
allowed to act.

Use this page before running unattended automation, file edits, shell commands,
or channel-connected agents.

For a focused permissions guide, see
[`approvals-and-permissions.md`](approvals-and-permissions.md).

## Built-In Tool Areas

| Area | Examples |
| --- | --- |
| Filesystem | `read_file`, `write_file`, `edit_file`, `list_dir`, `glob_search`, `grep_search`, spreadsheet reads. |
| Shell and code | `exec_command`, `background_process`, `process`, `execute_code`. |
| Git | `git_status`, `git_diff`, `git_log`, `git_commit`, `apply_patch`. |
| Web | `web_search`, `web_fetch`, `http_request`, `x_search` (X/Twitter via xAI — see [x-search.md](x-search.md); hidden without an xAI key). |
| Browser | `browser` — drive a real Chromium (navigate, snapshot, click, type, fill, wait, eval, dialog, screenshot) via the `agent-browser` engine. Hidden until the binary is installed. **Runs outside the sandbox** (Chromium can't run inside bubblewrap/seatbelt). See [features/browser.md](features/browser.md). |
| Memory | `memory_search`, `memory_save`, `memory_get`, `memory_delete`, `memory`. |
| Sessions | `sessions_send`, `sessions_spawn`, `sessions_list`, `sessions_history`, `session_status`, `session_rename`. `sessions_spawn` with `runtime="acp"` delegates to an external coding harness (Claude Code, Codex, ...); see [acp-agents.md](acp-agents.md). |
| ACP harnesses | `acp` — operator tool (not exposed by default) for harness sessions: `agents`, `sessions`, `status`, `cancel`, `close`, `set_mode`, `set_model`, `doctor`. Harnesses run as the gateway user, **outside the AgentOS sandbox**, with their own tools; their permission prompts follow `[acp] permission_mode`. |
| Artifacts | `publish_artifact`. The mime decides the rendering: some draw inline in chat rather than as a download chip — see [artifacts-and-media.md](artifacts-and-media.md#inline-charts). |
| Media | image generation, PDF, TTS, and media helpers. |
| Skills | `skill_list`, `skill_view`, `skill_create`, `skill_edit`, `install_skill_deps`. |
| Control | cron scheduling and gateway control operations. |
| Channels/platforms | messaging, chat, and media helpers across supported channel adapters. |
| User interaction | `ask_user` — structured questions with 2-4 options each. Presenting the question ends the turn; the answer arrives as the next user message. The Web UI renders a clickable card; the CLI and channels render a numbered list you answer by typing. Hidden on surfaces with nobody to reply (cron, subagents, heartbeats); channel DMs count as having a responder. |
| Plan mode | `/plan` flips the session into research-only planning: the turn's tool surface narrows to a read/search/analysis allowlist and the agent presents its finished plan via `exit_plan_mode` (which ends the turn, like `ask_user`). Approval is out of band — the Web UI plan card's Approve button or `/plan off` — so the model cannot talk itself out of plan mode. State is per-session gateway memory; a gateway restart clears it. |

## Permission Modes

Use stricter modes when running unattended:

```sh
agentos agent --permissions restricted -m "Inspect this repo"
```

Use broader modes only when you trust the task and workspace:

```sh
agentos agent --permissions full --workspace /path/to/project -m "Run tests and fix failures"
```

For interactive work, the Web UI approvals surface can pause sensitive tool
calls for review. For automation, choose a permission mode and workspace policy
before the run starts.

Read: [`approvals-and-permissions.md`](approvals-and-permissions.md)

## Approval Flow

Sensitive actions may pause for human approval depending on permission mode,
tool policy, channel surface, and runtime configuration.

Approvals are most important for:

- filesystem writes;
- shell commands;
- external channel or webhook delivery;
- generated artifacts that will be published;
- actions that affect another service.

Use the Web UI approvals page when you want durable review outside the chat
scrollback.

## Workspace Controls

Read-side restriction:

```sh
agentos agent --workspace /path/to/project --workspace-strict -m "Summarize this repo"
```

Write containment:

```sh
agentos agent \
  --workspace /path/to/project \
  --workspace-lockdown \
  --scratch-dir /path/to/project/.scratch \
  -m "Investigate and prepare a minimal patch"
```

`--workspace-lockdown` is intended for automation where writes must stay inside
the workspace or scratch directory.

For shell commands, lockdown inspects the command text for write targets —
redirections (`>`, `>>`, `2>`, `&>`, `>&`, `>|`, with or without surrounding
whitespace) and `tee` (including pipe-adjacent `|tee` and long options such as
`tee --append`). The scan reads text, not shell syntax, so it deliberately fails
closed: a command that merely *mentions* a path after `>` inside quotes — say
`grep '>/dev/null' script.sh` — is refused under lockdown even though it writes
nothing. That is defense in depth, not a security boundary: a
command can still reach outside the workspace through an interpreter
(`python -c`), a copy (`cp`, `dd of=`), an in-place edit (`sed -i`), or a path
built from a shell variable. Use the OS sandbox (`agentos sandbox on`) when you
need containment that does not depend on reading the command line.

## Sandbox Commands

```sh
agentos sandbox status
agentos sandbox on
agentos sandbox full
agentos sandbox bypass
agentos sandbox reset
```

Sandbox behavior is platform-dependent. Treat `sandbox status` and `doctor` as
the source of truth for the current machine.

## Recommended Patterns

| Task | Recommended posture |
| --- | --- |
| Read-only repo summary | `--workspace` plus `--workspace-strict` |
| Local patch with tests | `--workspace`, `--workspace-lockdown`, and a scratch dir |
| Chat with possible writes | Web UI with approvals visible |
| Channel-connected agent | Conservative permissions and explicit channel config |
| Provider/debug investigation | Diagnostics on, minimal tool permissions |

## Web Safety

AgentOS web tools use provider configuration and guardrails. Use provider
diagnostics when web search behaves unexpectedly:

```sh
agentos search status
agentos search query "test query"
agentos diagnostics on
```

Search results and fetched pages are external data. They should inform the
answer, not override tool policy or user instructions.

`http_request` refuses cloud metadata endpoints (`169.254.169.254`,
`metadata.google.internal`) on every configuration; ordinary private addresses
stay reachable, so pointing it at a local dev server still works. `web_fetch`
is stricter and blocks private addresses outright.

## Credentials

Authenticated API calls work normally: an API key in an `Authorization` or
`x-api-key` header is not refused. What outbound tools refuse is credential
*material* with no legitimate destination — a PEM private key, a
vendor-prefixed provider key (`sk-ant-…`, `ghp_…`, `AKIA…`), a connection
string with an inline password, an `/etc/passwd` line.

Prefer referencing a credential over pasting it. In a shell command use
`$NAME`; for sandboxed code, have the skill declare it:

```yaml
metadata:
  agentos:
    requires:
      env: [CAP_API_KEY]
```

The value is then read at spawn time and never enters the transcript. Command
output is masked before the model sees it, and the gateway token never reaches
a child process.

Read: [`configuration.md`](configuration.md#credentials-and-child-processes)

## Tool Compression

Large tool results may be compacted before they are shown to the model. This is
normal and protects the active context window. See
[`features/tool-compression.md`](features/tool-compression.md).

## Artifacts and Media

Tool calls can publish artifacts and generate media. See
[`artifacts-and-media.md`](artifacts-and-media.md) for user-facing artifact,
document, image, PDF, and TTS workflows.

## Troubleshooting

If a tool does not run:

1. Check permission posture:

   ```sh
   agentos sandbox status
   agentos doctor
   ```

2. Check whether the gateway or channel surface requires approval.
3. Confirm the workspace path is correct.
4. Use diagnostics for repeated failures:

   ```sh
   agentos diagnostics on
   ```

---

[Docs index](README.md) · [Product guide](../README.product.md) · [Improve this page](contributing-docs.md) · [Report a docs issue](https://github.com/use-agent-os/agent-os/issues/new?template=docs_report.yml)
