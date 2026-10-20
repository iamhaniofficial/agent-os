# Desktop App (macOS)

AgentOS App is the native desktop app for macOS 13 or newer, on Apple silicon
and Intel. It installs the engine for you, starts and supervises the local
gateway, and gives you chat and the Trading Desk in one window. It talks to
the same gateway as the [Web UI](web-ui.md) and the [CLI](cli.md), so
sessions, projects, approvals and scheduled jobs are shared between all
three.

## Install

1. Download the dmg for your Mac from the
   [latest release](https://github.com/use-agent-os/agent-os/releases/latest):
   `AgentOS-<version>-arm64.dmg` for Apple silicon, `AgentOS-<version>.dmg`
   for Intel. The app is signed with a Developer ID and notarized, so it
   opens without any Gatekeeper workaround.
2. Drag AgentOS into Applications and open it.
3. On first launch the app looks for an installed `agentos` CLI. If none is
   found (or it is older than the app), the setup screen offers **Install**:
   it installs the `use-agent-os` engine — the same package the terminal
   install uses — with no terminal, Python or `uv` needed. If a gateway is
   already running on the configured endpoint, the app connects to it
   instead; **Connect to an existing gateway instead** does the same for a
   gateway on another host.
4. Pick a provider. OpenCAP is listed first as **Recommended**; any provider
   in the catalog works. **Test key** checks the key against the provider
   before you save, and the last step tries it for real. **Skip for now**
   leaves a "Choose a provider" card on Home until one is configured.

The managed gateway's output goes to `~/Library/Logs/AgentOS/gateway.log`
(Settings › Advanced › App logs). Settings › Advanced also has **Reinstall
engine** and **Remove engine**; removing the engine keeps `~/.agentos`.

## Chat and Trade

The mode pill at the top of the window reads **Chat | Trade** (⌘⇧T switches).

- **Chat** is the conversation surface: streaming replies, tool activity,
  approvals, plan cards, `ask_user` question cards and artifacts — the same
  transcript the Web UI shows, in the desktop's own styling.
- **Trade** opens the Trading Desk: wallets, swaps, holdings, orders,
  history and allowances. See [Trading](features/trading.md).

The strip next to the pill shows **AWAITING** while a decision is waiting
for you and **RUNNING** while a mission is in flight.

### Run modes

The sliders button left of the paperclip in the composer opens **Run modes**
for the open session:

| Control | What it does |
| --- | --- |
| Execution | Approval prompts or bypass for this session. The global default is shown next to it. |
| Pilot Router | Turns [model routing](features/agentos-router.md) on or off for this session. |
| Plan mode | Research-only until you approve a plan (same as typing `/plan`; see [Plan mode](web-ui.md#plan-mode)). |
| Usage | The session's token usage so far. |

### Files and cards in the chat

A file the agent generates shows as a **file card**: a tile drawn by kind
(spreadsheet, document, PDF, presentation, archive, data, code, web page),
the file name, a subtitle such as "Spreadsheet · XLSX · 5 KB", and a
**Download** button. Images and audio play inline.

Trading commands run with `--json` publish cards the chat renders live:

- **DCA mandates** show a countdown to the next buy, spent-of-cap progress,
  average buy price vs now and a buys chart, with **Approve & start**,
  **Pause**, **Buy now** and **Stop** on the card. See [DCA](dca.md).
- **Uniswap V4 liquidity** cards show a pool, its range distribution or a
  position. See [LP cards](lp-cards.md) and [LP writes](lp-write.md).

## Sidebar

The sidebar lists your sessions, grouped by Today, Yesterday and This week,
with project folders above them.

- **Projects** are folders. Drag a session onto a folder to file it there
  (onto the "Sessions" header to unfile it). The `+` just left of a folder's
  chat count starts a new chat in that project with its agent; it appears on
  hover or keyboard focus. Click a folder to open its page, where the title
  renames on click and the shared brief saves itself (⌘S saves now). See
  [Projects](sessions.md#projects-group-sessions-and-share-knowledge).
- **Session names** come from the first message, whether the chat started
  here, in the Web UI or in a channel such as Telegram. A name you set is
  never overwritten. See [Session titles](configuration.md#session-titles).
- **Right-click a session** to rename, pin, mark as unread, export as
  Markdown, move it to a project, archive or delete it.

### Selecting several sessions

| Action | Effect |
| --- | --- |
| ⌘-click | Add or remove one row from the selection. |
| ⇧-click | Select the range from the last row clicked, in the order shown (chats inside open project folders included). |
| ⌘A | Select every row shown. |
| Escape | Clear the selection. |

Right-click inside the selection for one menu that acts on all of it:
**Pin**, **Mark as unread**, **Archive** and **Delete N sessions…** (asked
once). Rows the gateway could not delete stay selected and a toast gives
the count. A plain click still opens a chat and clears the selection.

## Scheduled jobs and skills

**Scheduled jobs** (⌘⇧J) opens as a panel over the current view, so checking
a schedule never leaves the conversation. Jobs use the same `cron.*` model
as the Web UI and `agentos cron`, with a natural-language schedule builder.
See [Scheduling](scheduling.md). **Skills** (⌘⇧K) browses and installs
skills; see [Skills](features/skills.md).

## Quick Ask

Press **⌥ Space** in any app to open a small prompt over it, the way
Spotlight works. Type a question and press **Return** to ask it in a new
chat, or **⌥ Return** to add it to the chat the window was on (or the last
one you opened). AgentOS comes forward on that chat with the reply
streaming. **Escape** or a click elsewhere closes the prompt without
sending; what you typed is still there next time.

Settings › Behaviour › Quick Ask turns the shortcut off or moves it to
⌃ Space or ⌘⇧ Space. If another app already uses the key, the pane says
"Unavailable"; choose another one. The prompt works while the app is
running, even with its window closed. A question asked while the gateway is
stopped waits in the chat and is sent once the gateway is running again.

## Notifications

The app notifies you when a reply finishes or fails, when an approval is
needed, when a scheduled job runs, and when the gateway stops on its own.
While you are watching the session in question it only plays the sound. The
toolbar bell keeps recent notifications, a Do not disturb switch (30 min,
1 h, 3 h, until tomorrow 9:00) and a sound toggle. Settings › Notifications
holds the per-event switches, the minimum reply length, the sound, and the
Dock badge and bounce. If nothing shows, check that AgentOS is allowed in
System Settings › Notifications and that no Focus mode is on.

## Settings

Open Settings with ⌘, or the toolbar gear. Escape or **Done** closes it.

| Group | Sections |
| --- | --- |
| Agent | Providers, Pilot Router, Skills, Channels, Environment, Trading |
| App | Gateway, Appearance, Notifications, Security, Behaviour, Shortcuts |
| More | Advanced, About |

Providers and Pilot Router edit the gateway's own configuration through the
same guided setup the Web UI uses. Appearance holds the theme
(system/light/dark), the colour palette, text size, reduced transparency and
the optional pet mascot. Behaviour holds open at login, what to open at
launch, stopping the gateway on quit, the Quick Ask shortcut, and Return vs
⌘Return to send.

Channels is the Web UI's Channels screen inside the app: every messaging
adapter the gateway is configured with (Telegram, Discord, Slack, Microsoft
Teams, Email), its live status, when it connected and how often it has been
restarted. **Add channel** opens a sheet built from the gateway's adapter
catalog: pick the adapter, paste what it needs, save; **Configure** reopens
it for an existing channel with the adapter and name locked, and a secret
left blank keeps the one already stored. Each card has an Enabled switch,
Restart and Remove, and a Telegram card lists who is waiting to pair and who
is in, with Pair, Deny and Disconnect. Changes to the configuration restart
a gateway the app manages so they apply at once; an external gateway is
told to restart.

Environment is the Web UI's Environment screen inside the app: every variable
the gateway, its providers and your skills read, grouped by what uses it,
with set, missing and shadowed counts. Set, replace, import from a source
that already holds the credential, or remove a variable in
`~/.agentos/.env`; **Add variable** stores one of your own. Listings never
carry a value. Showing a secret asks first, is rate limited by the gateway,
and hides again after 30 seconds.

## Updates

The app and the engine update together from one release.

- New releases show up as an **Update** pill in the toolbar and a single
  notice. Nothing downloads or installs until you click: **Update** updates
  the engine in place, then downloads the app; **Restart** installs it. The
  pill stays until the release is fully applied.
- Settings › About has an **Engine** card and an **App** card. Each shows
  the running version and a **Latest** row with the newest version ("Not
  checked yet" before the first check) and, while the app downloads, the
  download percent beside it.
- If an update would interrupt a live reply, the app sends you to
  Settings › About first, where **Update anyway** lives.
- After `agentos upgrade` in a terminal, the app offers **Restart gateway**
  so the running gateway picks up the new engine.

## Keyboard Shortcuts

The full list is in Settings › Shortcuts.

| Keys | Action |
| --- | --- |
| ⌥ Space (from any app) | Quick Ask (Settings › Behaviour) |
| ⌘N | New session |
| ⌘⇧O | New chat in this agent |
| ⌘, | Open settings |
| ⌘⇧K | Open skills |
| ⌘⇧J | Open scheduled jobs |
| ⌘⇧T | Switch Chat / Trade |
| ⌘⇧S | Show or hide the sidebar |
| ⌘+ / ⌘− / ⌘0 | Larger UI / smaller UI / actual size |
| Escape | Stop the current reply, or close a sheet |
| Return (or ⌘Return) | Send; the other key adds a new line (Settings › Behaviour) |
| ↑ / ↓ | Previous or next message you sent |
| / | Slash commands |
| ⌘S | Save a project brief now |

## Troubleshooting

- **"The gateway is not running."** Start it from the sidebar, or open
  Settings › Gateway for Start, Stop and Restart and the endpoint. See
  [Gateway](gateway.md).
- **Install failed.** The failure screen offers Retry, Copy output and Show
  log in Finder; the log is in `~/.agentos/logs/bootstrap-<timestamp>.log`.
- **Copy diagnostics** in Settings › Advanced copies a report with the
  gateway token redacted, for a bug report.

For building the app from source, see
[`desktop/README.md`](https://github.com/use-agent-os/agent-os/blob/main/desktop/README.md).

---

[Docs index](README.md) · [Product guide](../README.product.md) · [Improve this page](contributing-docs.md) · [Report a docs issue](https://github.com/use-agent-os/agent-os/issues/new?template=docs_report.yml)
