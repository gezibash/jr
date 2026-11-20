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
| Search by meaning | An embedding model. Default: [Ollama](https://ollama.com) on this machine with `ollama pull embeddinggemma-2:270m` |
| Reranking, captions, summaries | A chat model that takes images. Default, when `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are set: Gemma 4 on Cloudflare Workers AI |
| `search --like` (attachments that look or sound alike) | Cloudflare Workers AI |
| Video and audio attachments | `ffmpeg`; audio transcription uses `parakeet-mlx` through `uvx` (Apple silicon) |

The embedding and chat models can come from any service with an
OpenAI-compatible API: OpenAI, Vercel AI Gateway, OpenRouter, Cloudflare,
Ollama, LM Studio, vLLM and others. Give each a base URL, a key and a model:

```bash
JR_EMBED_URL=https://api.openai.com/v1  JR_EMBED_KEY=…  JR_EMBED_MODEL=text-embedding-3-small
JR_CHAT_URL=https://ai-gateway.vercel.sh/v1  JR_CHAT_KEY=…  JR_CHAT_MODEL=google/gemini-3-flash
JR_CHAT_EXTRA='{"reasoning_effort":"low"}'   # optional JSON fields added to each chat request
```

`jr doctor` shows which parts are on, which models they use, and how to turn
on the rest. After turning one on, or switching embedding model, `jr reindex`
indexes everything saved before. Vectors from different embedding models are
never compared.

A dedicated reranker with a Cohere-style `/rerank` API is faster than asking a
chat model (`JR_RERANK_URL`, `JR_RERANK_KEY`, `JR_RERANK_MODEL`; Cohere,
Voyage, Jina, or any of them through Vercel AI Gateway).

Measured on the same 20,000 entries and 30 reworded questions:

| Embeddings + reranker | Right entry first | Median / slowest 5% | Notes |
|---|---|---|---|
| EmbeddingGemma 2 (local) + Gemma 4 (Workers AI) | 22 of 30 | 0.8 s / 4.2 s | the default |
| EmbeddingGemma 2 (local) + Voyage rerank-3-lite | 22 of 30 | 0.56 s / 0.73 s | fastest |
| Qwen3-Embedding-4B (local) + Gemma 4 | 24 of 30 | 0.93 s / 1.8 s | 8.8 GB of memory in Ollama |
| Qwen3-Embedding-4B (Vercel) + Voyage rerank-3-lite | 25 of 30 | 1.3 s / 2.2 s | best answers |

Cohere Rerank 5 Pro and Fast and Voyage rerank-3 scored 20 to 22, like Gemma:
for most of the misses, the right entry was never among the candidates, so
better embeddings help more than a better reranker.

The numbers below were measured with EmbeddingGemma 2 and Gemma 4. Other
models will score differently; the reranking and summary prompts were tuned
on Gemma 4. Keep thinking short or off: a model that reasons at length can
use up its reply budget and return nothing.

Measured on 20,000 entries spread over 3 months, with 30 test questions:

| | No setup | With Ollama and Workers AI |
|---|---|---|
| Question with the entry's own words: right entry first | 30 of 30 | 30 of 30 |
| Question in other words: right entry first | 8 of 30 | 21 of 30 |
| Search time, median / slowest 5% | 0.06 s / 0.07 s | 1.1 s / 4.2 s |
| Adds per second, 16 sessions at once | 156 | 58 |

The reworded questions avoid the entry's words on purpose, so they are the
hard case. Search by meaning is what closes that gap.

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
