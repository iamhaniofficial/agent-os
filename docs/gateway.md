# Gateway

The AgentOS gateway is the local server behind the Web UI, channels, RPC
clients, sessions, approvals, diagnostics, and usage views. Most day-to-day
AgentOS surfaces work best when the gateway is running.

Use this page when you want to start, stop, inspect, expose, or troubleshoot
the gateway.

## Foreground Gateway

Run the gateway in the current terminal:

```sh
agentos gateway run
```

Open the control console:

```text
http://127.0.0.1:18791/control/
```

Stop a foreground gateway with `Ctrl+C`.

## Managed Background Gateway

Start a managed background process and wait for readiness:

```sh
agentos gateway start --json
```

Inspect it:

```sh
agentos gateway status
agentos gateway status --json
```

Restart or stop it:

```sh
agentos gateway restart
agentos gateway stop
```

Use the managed gateway for the Web UI, channels, scheduled jobs, and local
automation that should survive the current terminal tab.

## Host and Port

Use a different port:

```sh
agentos gateway run --port 18792
agentos gateway status --port 18792
```

Bind to a specific host:

```sh
agentos gateway run --listen 127.0.0.1 --port 18791
```

`--listen` is an alias for the bind host and wins over `--bind` when both are
provided.

## Safety Defaults

The gateway defaults to loopback scope, usually `127.0.0.1`, because the local
gateway controls chat, tools, sessions, channels, approvals, and configuration.

Public binding is opt-in:

```sh
agentos gateway run --listen 0.0.0.0 --port 18791
```

Do not expose a gateway to an untrusted network without token auth and a network
boundary you understand.

## Configuration Path

Use a specific config file:

```sh
agentos gateway run --config /path/to/agentos.toml
agentos gateway status --config /path/to/agentos.toml
```

AgentOS also reads standard configuration locations described in
[`configuration.md`](configuration.md).

## Remote Status Check

Inspect a gateway URL directly:

```sh
agentos gateway status --gateway ws://localhost:18791/ws
```

This is useful when a client or MCP bridge is configured with an explicit
gateway URL.

## When to Restart

Restart the gateway after changing:

- provider or router configuration;
- channel configuration;
- durable agent entries;
- global sandbox posture;
- search or image-generation setup;
- environment variables used by configured providers.

```sh
agentos gateway restart
```

That works for a gateway started with `agentos gateway start` (stop, then
start) and for one started any other way — the desktop app, launchd, systemd
or a terminal `agentos gateway run` — which restarts itself **in place**: the
gateway drains the turns in flight, re-executes itself under the same process
id so every launcher keeps tracking it, and comes back on upgraded code. The
CLI confirms the restart from the gateway's readiness probe. Not available on
Windows, where the process image cannot be replaced.

The same in-place restart is reachable from chat:

- `/restart` in the web console or the desktop app (operators only, no
  setting needed). The socket drops and reconnects; a toast reports when the
  gateway is back.
- `/restart` from a paired Telegram, Discord or Slack sender, and the agent's
  own `gateway` tool, once `[gateway_control] chat_restart = true` is set and
  the gateway has been restarted once. The reply acknowledges, then "Gateway
  restarted (v…, Ns)" lands in the same chat or thread after boot.

`/restart` waits up to 30 seconds for active turns to finish; `/restart force`
(or `agentos gateway restart --force`) cancels them instead. A second
request while one is pending is reported, not queued.

## Troubleshooting

Check status and readiness:

```sh
agentos gateway status
agentos doctor
```

If the port is busy:

```sh
agentos gateway run --port 18792
```

If the Web UI cannot connect, confirm that the URL matches the gateway bind
host and port.

Read next:

- [`http-api.md`](http-api.md)
- [`web-ui.md`](web-ui.md)
- [`configuration.md`](configuration.md)
- [`channels.md`](channels.md)
- [`troubleshooting.md`](troubleshooting.md)

---

[Docs index](README.md) · [Product guide](../README.product.md) · [Improve this page](contributing-docs.md) · [Report a docs issue](https://github.com/use-agent-os/agent-os/issues/new?template=docs_report.yml)
