# code-search-mcp

Local semantic code search for coding agents, powered by Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). One MCP server works in
**Codex**, **Claude Code** (CLI, IDE, desktop Code tab) and **Claude Desktop**. Everything runs on
your machine: code, index and model.

## Tools

| Tool | What it's for |
|---|---|
| `search_code` | Find code by meaning or name. Hybrid vectors + BM25, symbol-aware results, optional reranker, multi-repo |
| `related_files` | Before editing: files that change with this one (git co-change + callers/callees + similarity) |
| `find_tests` | Tests covering a file, `file:line` or symbol |
| `find_similar` | Duplicated logic, or an existing pattern to follow |
| `search_history` | "When/why did we change X?" over commit messages and changed functions |
| `explain_area` | Map of an unfamiliar repo: functional areas, core files, key symbols |
| `search_by_image` | Screenshot or mockup → the code behind it |
| `index_repo`, `index_status` | Manual index control and status (indexing is normally automatic) |

How it stays fast and fresh:

- **Tree-sitter chunking**: chunks follow real function, class and method boundaries across most languages,
  and each chunk records its qualified symbol name and signature. Prose and config use a heading/line splitter.
- **Background watcher**: re-indexes a few seconds after you save. It watches only directories with
  source files, never `node_modules` or build output.
- **Embedding cache by content**: switching branches, rebasing or reverting re-uses earlier embeddings,
  so going back to a branch costs no model time.
- **Never blocks the agent**: a first index of a big repo runs in the background, and tools answer
  from the partial index with a progress note.
- **Safe with several clients at once**: Codex and Claude Code can run it side by side on the same repo.
- The index lives in `~/.cache/code-embed/`, nothing is written into your repo, and `.gitignore` is respected.

## Install

Requires [uv](https://docs.astral.sh/uv/). Put this folder somewhere permanent, e.g. `~/tools/code-search-mcp`, then:

```bash
cd ~/tools/code-search-mcp
uv sync                                   # installs deps (PyTorch etc.) into .venv
uv run code-search index ~/code/my-repo   # optional: pre-build; first run downloads the model (~1.5 GB)
uv run code-search search "where are JWTs validated" --path ~/code/my-repo
```

The commands below launch the server with `uv run --project ~/tools/code-search-mcp code-search serve`.
Use absolute paths; in desktop apps also use the full path to `uv` (`which uv`).

### Codex

```bash
codex mcp add code-search -- uv run --project ~/tools/code-search-mcp code-search serve
```

or in `~/.codex/config.toml`:

```toml
[mcp_servers.code-search]
command = "uv"
args = ["run", "--project", "/Users/you/tools/code-search-mcp", "code-search", "serve"]
startup_timeout_sec = 120
tool_timeout_sec = 300
default_tools_approval_mode = "writes"   # tools are read-only, so no approval prompts

[mcp_servers.code-search.env]
# CODE_EMBED_DEVICE = "mps"              # see Configuration
```

### Claude Code (CLI, IDE, and the Code tab in Claude Desktop)

```bash
claude mcp add --scope user --transport stdio code-search -- \
  uv run --project ~/tools/code-search-mcp code-search serve
```

Claude Code tells the server which project is open, so tools default to the right repo. To share
with a team, use `--scope project`, which writes `.mcp.json`:

```json
{
  "mcpServers": {
    "code-search": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--project", "/Users/you/tools/code-search-mcp", "code-search", "serve"],
      "timeout": 300000
    }
  }
}
```

### Claude Desktop (chat)

Settings → Developer → Edit Config (`claude_desktop_config.json`), then restart the app:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "/Users/you/.local/bin/uv",
      "args": ["run", "--project", "/Users/you/tools/code-search-mcp", "code-search", "serve"],
      "env": { "CODE_EMBED_REPOS": "/Users/you/code/api:/Users/you/code/web" }
    }
  }
}
```

Chat has no "current project": name the repo in your message, or set `CODE_EMBED_REPOS` and ask
it to search `repos: ["all"]`. On Windows, separate repo paths with `;`.

### Tell the agent to use it

Add to `AGENTS.md` (Codex) or `CLAUDE.md` (Claude Code):

```markdown
## Code search
Use the code-search MCP tools before grep/find when you don't know where something lives:
search_code to locate code by meaning, related_files before editing a file, find_tests to find
covering tests, find_similar before writing a new helper, search_history for "why did this change".
```

## Configuration

All settings are environment variables. Put them in the `env` block of your client config.

| Variable | Default | |
|---|---|---|
| `CODE_EMBED_DEVICE` | auto | `cuda`, `mps` (Apple Silicon) or `cpu` |
| `CODE_EMBED_DIM` | `768` | `512` / `256` / `128`: smaller, faster index (Matryoshka truncation) |
| `CODE_EMBED_RERANK` | off | Cross-encoder for sharper ranking, e.g. `BAAI/bge-reranker-v2-m3` |
| `CODE_EMBED_REPOS` | none | Repos to watch from startup and include in `repos: ["all"]` |
| `CODE_EMBED_IMAGES` | `0` | `1` = also index png/jpg/webp in repos (for `search_by_image`) |
| `CODE_EMBED_WATCH` | `1` | `0` = no background watcher (refresh on each search instead) |
| `CODE_EMBED_HISTORY` | `3000` | How many recent commits `search_history` indexes |
| `CODE_EMBED_MAX_LINES` | `60` | Max lines per chunk |
| `CODE_EMBED_MODEL` | `google/embeddinggemma-2` | Any sentence-transformers model or a fine-tuned folder |
| `CODE_EMBED_BACKEND` | `st` | `st` (in-process) or `http` (see below) |
| `CODE_EMBED_CACHE` | `~/.cache/code-embed` | Where indexes live |

Changing the model or dimension creates a separate index, so you can compare setups side by side.

### Lighter and faster: quantized GGUF via llama.cpp

Run the model in [llama-server](https://unsloth.ai/docs/models/embeddinggemma-2) (Unsloth's 4-bit
GGUF) and point the server at it. This also works with Ollama or LM Studio's OpenAI-compatible endpoint.

```bash
hf download unsloth/embeddinggemma-2-GGUF embeddinggemma-2-UD-Q4_K_XL.gguf --local-dir gguf
llama-server --model gguf/embeddinggemma-2-UD-Q4_K_XL.gguf --alias embeddinggemma2 \
  --embeddings --pooling mean --ctx-size 2048 --batch-size 2048 --ubatch-size 2048 \
  --host 127.0.0.1 --port 8080
```

```
CODE_EMBED_BACKEND=http
CODE_EMBED_URL=http://127.0.0.1:8080/v1/embeddings
CODE_EMBED_HTTP_MODEL=embeddinggemma2
```

The server sends the model-card prompts itself. Images need the in-process `st` backend.

## Measure, then fine-tune on your repo

Build an eval set from your own git history (commit message → files it changed), then measure:

```bash
uv run code-search eval-generate eval.jsonl --path ~/code/my-repo -n 150
uv run code-search eval eval.jsonl --path ~/code/my-repo -v        # recall@1/3/5/10, MRR
CODE_EMBED_DIM=256 uv run code-search eval eval.jsonl --path ~/code/my-repo   # compare a setting
```

You can also hand-write eval lines: `{"query": "...", "expected": ["path", "path:42", "path::Symbol"]}`.

Fine-tune EmbeddingGemma 2 on pairs mined from the repo (docstring → code, commit message → changed
functions). This needs a GPU in practice.

```bash
uv sync --extra finetune
uv run code-search finetune ./my-repo-embed --path ~/code/my-repo --lora   # prints held-out MRR before/after
CODE_EMBED_MODEL=$PWD/my-repo-embed uv run code-search eval eval.jsonl --path ~/code/my-repo
```

Training uses bf16 on capable GPUs and fp32 otherwise. EmbeddingGemma 2 must never run in fp16:
it silently produces NaN or degraded vectors, and the server refuses NaN embeddings. If the
fine-tuned model scores better, set `CODE_EMBED_MODEL` to the folder in your client config. A
`--lora` output is a small adapter on top of the base model.

## Terminal commands

`code-search index | search | history | related | tests | explain | status | eval | eval-generate | finetune`
(`--help` on each). `code-search serve` runs the MCP server.

## Development

```bash
uv sync --extra dev
CODE_EMBED_BACKEND=hash uv run pytest -q    # offline test backend, no model download
```

Works with MCP Python SDK 1.x and 2.x.
