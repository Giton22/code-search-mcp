# code-search-mcp

Local semantic code search for coding agents, powered by Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). One MCP server works in
**Codex**, **Claude Code** (CLI, IDE, desktop Code tab) and **Claude Desktop**. Everything runs on
your machine: code, index and model.

## Tools

| Tool | What it's for |
|---|---|
| `prepare_context` | Start here for a task: the most relevant code in full, what it calls, who uses it, its tests and related commits, packed into a token budget. Every item has a handle for `expand` |
| `expand` | Full code for a handle: `path:12-40`, `path:12`, `path::Symbol`, a symbol, or a file |
| `find_definition`, `find_references` | Where a symbol is defined, and every place that uses it (code graph from the tree-sitter index) |
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
- **One shared model**: the first client starts a background service that loads the model once.
  Every Codex/Claude window, terminal command and the UI use it, so a second window costs no extra
  GPU memory or load time. It exits after 30 idle minutes. `CODE_EMBED_DAEMON=0` keeps everything
  in-process instead.
- **Safe with several clients at once**: Codex and Claude Code can run it side by side on the same repo.
- The index lives in `~/.cache/code-embed/`, nothing is written into your repo, and `.gitignore` is respected.

## Install

Requires [uv](https://docs.astral.sh/uv/). Put this folder somewhere permanent, e.g. `~/tools/code-search-mcp`, then:

```bash
cd ~/tools/code-search-mcp
uv run code-search setup                  # installs everything, with the PyTorch build for your GPU
uv run code-search ui                     # optional: the web UI (search, repos, GPU, agent setup)
uv run code-search index ~/code/my-repo   # optional: pre-build; first run downloads the model (~1.5 GB)
uv run code-search search "where are JWTs validated" --path ~/code/my-repo
```

`setup` picks the GPU automatically:

| Hardware | What it does |
|---|---|
| NVIDIA (Windows, Linux) | Installs the CUDA build that matches your driver (`uv --torch-backend auto`) |
| AMD Radeon (Windows, Linux) | Installs AMD's ROCm build for your exact GPU (e.g. gfx1201 for RX 9070 XT); no ROCm SDK needed |
| Apple Silicon | Installs the standard build, which runs on the Apple GPU (MPS) |
| Intel Arc | Installs the XPU build |
| None of these | Installs the CPU build |

`uv run code-search setup --dry-run` shows what it would install. At runtime the server picks the
best device on its own (a discrete GPU over an integrated one) and logs it, and `index_status`
reports it. PyTorch is left out of uv's lockfile on purpose, so `uv run` never swaps your GPU build
for another one. A plain `uv sync` does remove it; if that happens, the server says so: run `setup` again.

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
# CODE_EMBED_DEVICE = "cuda:1"           # see Configuration; normally auto-detected
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
prepare_context at the start of a task, expand to read a handle in full, search_code to locate code
by meaning, find_references before changing a signature, related_files before editing a file,
find_tests to find covering tests, find_similar before writing a new helper, search_history for
"why did this change".
```

### Automatic context on every prompt (Claude Code)

```bash
uv run code-search hook install                 # ~/.claude/settings.json; --scope project for ./.claude
uv run code-search hook uninstall
```

This adds a `UserPromptSubmit` hook. On each prompt it attaches the few code locations that match
confidently (as `path:lines` handles), so the agent starts in the right place even when it forgets
to search. It skips short replies and slash commands, adds nothing when no match clears the
threshold, never waits for a model to load, and gives up after 8 seconds.

## Configuration

All settings are environment variables. Put them in the `env` block of your client config.

| Variable | Default | |
|---|---|---|
| `CODE_EMBED_DEVICE` | auto | Override the detected device: `cuda` / `cuda:1` (NVIDIA and AMD), `mps`, `xpu` or `cpu` |
| `CODE_EMBED_DIM` | `768` | `512` / `256` / `128`: smaller, faster index (Matryoshka truncation) |
| `CODE_EMBED_RERANK` | `Qwen/Qwen3-Reranker-0.6B` with a GPU, off on CPU | Reranker that re-sorts the top results. `off` disables it. Others: `jinaai/jina-reranker-v3` (ranked best on the benchmark, but non-commercial license and needs `CODE_EMBED_TRUST_REMOTE_CODE=1`), `Alibaba-NLP/gte-reranker-modernbert-base`, `BAAI/bge-reranker-v2-m3` |
| `CODE_EMBED_RERANK_TOP` | `30` | How many top results the reranker re-scores |
| `CODE_EMBED_GPU_MEMORY` | `0.5` | Most of the GPU's memory the model may use (fraction); `1` = no cap. Indexing also runs at below-normal CPU priority, so the machine stays usable |
| `CODE_EMBED_REPOS` | none | Repos to watch from startup and include in `repos: ["all"]` |
| `CODE_EMBED_IMAGES` | `0` | `1` = also index png/jpg/webp in repos (for `search_by_image`) |
| `CODE_EMBED_WATCH` | `1` | `0` = no background watcher (refresh on each search instead) |
| `CODE_EMBED_HISTORY` | `3000` | How many recent commits `search_history` indexes |
| `CODE_EMBED_MAX_LINES` | `60` | Max lines per chunk |
| `CODE_EMBED_MODEL` | `google/embeddinggemma-2` | Any sentence-transformers model |
| `CODE_EMBED_BACKEND` | `st` | `st` (in-process) or `http` (see below) |
| `CODE_EMBED_CACHE` | `~/.cache/code-embed` | Where indexes live |
| `CODE_EMBED_DAEMON` | `1` | `0` = load the model in each client process instead of the shared service |
| `CODE_EMBED_DAEMON_IDLE_MINUTES` | `30` | When the shared service exits after its last request |
| `CODE_EMBED_HOOK_MIN_SCORE` | `0.5` | How confident a match must be before the prompt hook attaches it |

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

## Benchmark

```bash
uv run code-search bench --variants hybrid,vector,bm25,context,rerank --out bench.json
```

Takes real GitHub issues from SWE-bench Lite and SWE-bench Multilingual, checks each repo out at the
commit before the fix, searches with the issue text and scores whether the files the fix changed
rank near the top (hit@1/5/10, recall@10, MRR, per language). Repos are fetched shallowly into the
cache and indexed separately from your own. `--repos` and `--per-repo` set the size.

## Measure on your repo

Build an eval set from your own git history (commit message → files it changed), then measure:

```bash
uv run code-search eval-generate eval.jsonl --path ~/code/my-repo -n 150
uv run code-search eval eval.jsonl --path ~/code/my-repo -v        # recall@1/3/5/10, MRR
CODE_EMBED_DIM=256 uv run code-search eval eval.jsonl --path ~/code/my-repo   # compare a setting
```

You can also hand-write eval lines: `{"query": "...", "expected": ["path", "path:42", "path::Symbol"]}`.

## Terminal commands

`code-search setup | ui | hook | index | search | context | expand | def | refs | history | related | tests | explain | status | bench | eval | eval-generate`
(`--help` on each). `code-search serve` runs the MCP server.

`code-search ui` opens a local web app in your browser: search with code previews (Enter opens the
result in VS Code, Cursor or Zed), history, related files, tests, a repo map, indexed repos with live
progress, the device in use, and copy-ready agent configs. It listens on 127.0.0.1 only.

## Development

```bash
uv sync --extra dev
uv run code-search setup                    # PyTorch isn't in the lockfile; only needed for the real model
CODE_EMBED_BACKEND=hash uv run pytest -q    # offline test backend, no model download
```

Works with MCP Python SDK 1.x and 2.x.
