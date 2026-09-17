---
name: cron-watchers
description: "Ready-made watcher scripts for cron script jobs — poll an RSS feed, a JSON endpoint, or a GitHub repo on a schedule and report only what is new, without spending a model call. Use when the user wants to be told about new items from a feed/API/repo, or wants a scheduled check that stays quiet until something changes."
always: false
triggers:
  - watch
  - watcher
  - monitor
  - poll
  - feed
  - rss
  - notify me when
  - theo dõi
provenance:
  origin: agentos-original
  license: MIT
metadata:
  agentos:
    requires_tools:
      - cron
---

# Cron watchers

Three scripts that answer "tell me when something new shows up" without a model
in the loop. Each one polls a source, remembers what it has already reported,
prints a line per new item, and prints **nothing** when there is nothing new —
which is exactly the contract a cron `script` job wants.

| Script | Watches | Key arguments |
| --- | --- | --- |
| `watch_rss.py` | An RSS or Atom feed | `--url`, `--name` |
| `watch_http_json.py` | A JSON endpoint returning a list | `--url`, `--name`, `--id-field`, `--items-path`, `--field`, `--header` |
| `watch_github.py` | A repo's issues, pulls, or releases | `--repo`, `--scope`, `--name` |

All three also take `--limit` (max lines per run, default 10) and
`--first-run-reports` (report on the first run instead of starting quiet).
When more than `--limit` items are new, the run reports the oldest of them and
leaves the rest for the following runs; only what was reported is marked seen.
An item is lost only if it drops off the feed's page before its turn comes.

`--url` must be `http://` or `https://`. Any other scheme is refused with exit
code 1 — `urlopen` speaks `file:`, `ftp:` and `data:` too, and a watcher
pointed at `file:///etc/passwd` would report the file's contents on every run.

## Install them first

A cron job may only run files under `~/.agentos/scripts/`, so copy the ones you
need there before scheduling:

```sh
mkdir -p ~/.agentos/scripts
cp {baseDir}/scripts/_watermark.py {baseDir}/scripts/_url.py ~/.agentos/scripts/
cp {baseDir}/scripts/watch_rss.py ~/.agentos/scripts/
```

`_watermark.py` (shared state-keeping) and `_url.py` (the `http(s)` check on
`--url`) are used by all three — copy both alongside whichever watcher you
install, or it will fail to start with an `ImportError`.

## Schedule one

Deliver the new items straight to the user, no LLM call:

```sh
agentos cron add --every 15m --name hn-watch --script watch_rss.py \
  --script-arg --name --script-arg hn \
  --script-arg --url --script-arg https://news.ycombinator.com/rss
```

Or let an agent read the findings and decide what is worth saying — the turn is
skipped entirely on ticks where the watcher printed nothing:

```sh
agentos cron add --every 10m --name repo-triage \
  --job-kind agent_turn \
  --script watch_github.py \
  --script-arg --repo --script-arg owner/name \
  --script-arg --scope --script-arg issues \
  --text "Summarize anything here that looks urgent. Stay brief."
```

From the `cron` tool, the same two shapes are `job_kind='script'` and
`job_kind='agent_turn'` with `script` plus `script_args`. Either way, scheduling
a script requires an interactive CLI or Web caller — it will be refused from a
chat channel.

## Behaviour worth knowing

- **The first run is silent by default.** A watcher that has never run cannot
  tell which of the 30 items on the page are new, so it records them and reports
  nothing. Pass `--first-run-reports` if you want the initial dump.
- **State lives outside the scripts directory**, in
  `~/.agentos/state/cron-watchers/<name>.json`. Use a distinct `--name` per
  source or two watchers will suppress each other's items.
- **A non-zero exit is delivered as an alert**, so a feed that starts returning
  502s surfaces instead of going quiet.
- **`watch_http_json.py` refuses an `--id-field` that matches nothing.** When
  items were fetched and none of them carries the field, the run exits 1 and
  names the field on stderr instead of reporting an empty feed; nothing is
  recorded, so the corrected run still counts as the first.
- `watch_github.py` reads `GITHUB_TOKEN` when it is set, for rate limits and
  private repos.

## Writing your own

Any executable that follows the same contract works: print what matters, print
nothing when nothing matters, exit non-zero on a real failure. A last line of
`{"wakeAgent": false}` is also read as "nothing to report" for scripts ported
from other runtimes. See `docs/scheduling.md` for the full contract.
