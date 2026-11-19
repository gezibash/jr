# jr

A journal per project, for coding agents and the people working with them.
Agents write down decisions, bug root causes, findings and screenshots as
they work, then search them later: "what did we do yesterday", "have we seen
this error before", "what's still open", "summarise last week".

- Entries live in SQLite, one journal per git remote (`~/.journal/<project>/`).
  Many sessions can write at once.
- Search mixes keywords and meaning, then reranks. Dates are local calendar
  days: `--on yesterday`, `--since monday`, `--on "last week"`.
- Attachments (images, video, audio) are captioned or transcribed, so search
  finds them by what they show or say.
- Summaries cite the entries they come from; long periods cite day summaries.

## Install

```bash
mise use -g github:gezibash/jr
```

Or download a binary from the releases page, or `go install github.com/gezibash/jr@latest`.

## Setup

None. Install the binary and start writing: entries, filters, dates,
open/closed tracking and keyword search all work with nothing else.

Each optional part adds more. Turn them on in any order, whenever you like:

| Adds | Needs |
|---|---|
| Search by meaning | [Ollama](https://ollama.com) with `ollama pull embeddinggemma-2:270m` |
| Reranking, captions, summaries, `search --like` | Cloudflare Workers AI: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| Video and audio attachments | `ffmpeg`; audio transcription uses `parakeet-mlx` through `uvx` (Apple silicon) |

`jr doctor` shows which parts are on and how to turn on the rest. After
turning one on, `jr reindex` indexes everything saved before.

## Use

```bash
jr add --kind bug --tag auth "Users logged out after token refresh

Root cause: the refresh handler raced with the session writer."
jr search "users logged out"
jr list --on yesterday
jr list --status open --on "last week"
jr summarize --on "last week"
jr help
```
