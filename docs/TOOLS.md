# AlphaAI Tools

Tools are the primitives AlphaAI (and its models) can call. Every tool call goes
through the **tool engine**, which enforces permissions, sandboxes file access,
applies timeouts and writes an audit log.

```shell
alphaai tools list          # registered tools + permission state
alphaai tools run calculator.evaluate '{"expression": "6*7"}'
alphaai tools calls         # recent tool-call audit log
```

## Built-in tools

| Tool | Purpose | Permissions |
| ---- | ------- | ----------- |
| `calculator.evaluate` | exact arithmetic via AlphaAI's safe evaluator | none |
| `file.read` | read a file inside the sandbox root | `filesystem.read` |
| `file.write` | write a file inside the sandbox root | `filesystem.write` |
| `file.list` | list a directory inside the sandbox root | `filesystem.read` |
| `text.transform` | deterministic string operations | none |
| `tool.parse_call` | parse a tool call out of model text | none |
| (others) | see the registry in `alphaai/core/tools/builtin.py` | per tool |

Tools are declared with an id, description, a JSON-schema parameter block, the
permissions they imply and a real handler. Model-facing descriptions are rendered
into the prompt by the engine, so a model can emit a call in a stable JSON form.

## Permissions

Permissions are configuration, not guesses. Defaults deny network and write
access until explicitly enabled in `configs/alphaai.toml`. A denied call returns
`tool_permission_denied` with a remediation — it never silently degrades.

| Permission | Default | Governs |
| ---------- | ------- | ------- |
| `filesystem.read` | allowed (sandboxed) | reading files under the sandbox root |
| `filesystem.write` | `false` | writing files under the sandbox root |
| `network` | `false` | any outbound request (e.g. web research) |
| `process` | `false` | spawning subprocesses |

## Sandbox

`tools.sandbox_root` (default `./workspace`) bounds every file path. Paths are
resolved and checked so a tool can never read or write outside the sandbox. Both
`file.write` and the skill layer reuse the same resolver.

## Audit log

Every call records the tool id, arguments hash, outcome, duration and run id.
`alphaai tools calls` and `GET /api/tools/calls` show it, which makes tool use
inspectable rather than opaque.

## Calling a tool over the API

```shell
curl -s localhost:8000/api/tools/execute \
  -H 'content-type: application/json' \
  -d '{"tool_id": "calculator.evaluate", "arguments": {"expression": "6*7"}}'
```

The response is `{"ok": <bool>, "result": {...}}`, where a failed result carries a
structured `error` with a `code`.
