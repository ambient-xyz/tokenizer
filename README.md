# tokenizer

Production-grade, async token counting utilities for Qwen2.5-0.5B-Instruct using a bundled tokenizer config (`qwen2_5_0_5b_instruct.json`) and the Hugging Face chat template (`qwen2_5_0_5b_instruct_chat_template.jinja`) that SGLang uses for `Qwen/Qwen2.5-0.5B-Instruct`.

This crate provides two entry points:

- `qwen(...)`: tokenize raw text (no chat template).
- `qwen_chat(...)`: tokenize chat messages after rendering the Qwen chat template (includes any extra template/prompt tokens).

## What this is for

- Fast, consistent token counting in services that need to:
  - enforce input/output token limits
  - price requests
  - validate prompts before sending to an LLM
- Qwen-specific behavior: the chat token count depends on template formatting, so `qwen_chat` gives a closer estimate to what the model receives.

## Features

- **Bundled assets**: ships with `qwen2_5_0_5b_instruct.json` tokenizer config and `qwen2_5_0_5b_instruct_chat_template.jinja` chat template embedded at compile time.
- **Async-friendly**: tokenization work is offloaded via `async-threadpool` to avoid blocking async runtimes.
- **Template compatibility**: uses `minijinja` plus `minijinja-contrib` `pycompat` callback to support common Jinja/Python-style methods (e.g. `strip`) used by templates.
- **Small API surface**: minimal types and functions; easy to integrate.

## Install

This crate is currently marked as `license = "Proprietary"` and is intended for internal/private consumption.

Add to your workspace / project:

```toml
[dependencies]
tokenizer = { path = "../tokenizer" }
````

If you consume it from Git:

```toml
[dependencies]
tokenizer = { git = "https://github.com/ambient-xyz/tokenizer", tag = "v0.2.6" }
```

> Note: `edition = "2024"`; ensure your toolchain supports Rust 2024 edition.

## Quick start

### Raw tokenization (no chat template)

```rust
use tokenizer::qwen;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let n = qwen("Hello, world!").await?;
    println!("raw tokens: {n}");
    Ok(())
}
```

### Chat tokenization (template applied)

```rust
use tokenizer::{qwen_chat, ChatMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let messages = vec![
        ChatMessage::system("You are a helpful assistant.".to_string()),
        ChatMessage::user("Hello, world!".to_string()),
    ];

    let n = qwen_chat(messages).await?;
    println!("chat tokens: {n}");
    Ok(())
}
```

## API

### `ChatMessage`

A serializable message type that matches typical chat role/content pairs.

```rust
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
```

Helpers:

* `ChatMessage::system(content: String)`
* `ChatMessage::user(content: String)`
* `ChatMessage::assistant(content: String)`
* `ChatMessage::tool(content: String)`

### `qwen(input) -> Result<usize, Error>`

Tokenizes raw input using the bundled Qwen tokenizer config.

* **Does not** apply any chat template.
* Generic over `E: Into<InputSequence<'a>> + Send + 'static` (e.g. `&str`, `String`).

### `qwen_chat(messages) -> Result<usize, Error>`

* Renders `qwen2_5_0_5b_instruct_chat_template.jinja` using MiniJinja with `pycompat`.
* Sets `add_generation_prompt => true` when rendering (so the final formatted prompt includes the model’s expected generation prompt).
* Tokenizes the rendered prompt using the bundled Qwen tokenizer config.

The old `glm` and `glm_chat` functions remain as deprecated compatibility aliases for existing callers.

## Error handling

All public APIs return:

```rust
Result<usize, tokenizer::Error>
```

Error variants:

* `Error::Tokenization(tokenizers::tokenizer::Error)` — tokenizer encode failures
* `Error::ThreadPool(async_threadpool::Error)` — threadpool execution failures
* `Error::Template(minijinja::Error)` — template parse/render failures

## Performance notes

* Tokenization is executed in an async threadpool to keep async request handlers responsive.
* The Qwen tokenizer is initialized once and cached globally using `std::sync::LazyLock`.
* `qwen_chat` creates a MiniJinja `Environment` per call; if you need ultra-high throughput, consider extending the crate to cache a prepared `Environment` + compiled template (thread-safe strategy required).

## Determinism and compatibility

* `qwen_chat` output is sensitive to the embedded template and the `add_generation_prompt` flag.
* Changing `qwen2_5_0_5b_instruct.json` or `qwen2_5_0_5b_instruct_chat_template.jinja` will change token counts; treat those as versioned, user-visible behavior.

## Testing

Run the full test suite:

```bash
cargo test
```

The tests include:

* tokenizer initialization
* raw token count sanity checks
* chat-vs-raw comparisons to ensure template overhead is present

## Repository layout

```
.
├── Cargo.toml
├── Cargo.lock
├── qwen2_5_0_5b_instruct.json
├── qwen2_5_0_5b_instruct_chat_template.jinja
└── src
    └── lib.rs
```

## Security and operational notes

* Inputs are treated as untrusted text. Tokenization and template rendering should not execute code, but very large inputs can consume CPU/memory. Enforce reasonable size limits at the API boundary.
* If you accept user-provided message lists, cap:

    * number of messages
    * total character length
    * maximum per-message length

## Versioning

This crate follows semantic versioning for its public API surface. Asset changes (`qwen2_5_0_5b_instruct.json`, template) may change outputs and should be considered behavior changes; prefer bumping at least the minor version when those are updated.

Current version: `0.2.6`

## Git LFS (Large Files)

This repo uses Git LFS for large tokenizer JSON files.

### Install Git LFS
- macOS (Homebrew):
  ```bash
  brew install git-lfs
  git lfs install
  ````

* Ubuntu/Debian:

  ```bash
  sudo apt update
  sudo apt install git-lfs
  git lfs install
  ```

### Clone and fetch LFS files

```bash
git clone <REPO_URL>
cd <REPO_DIR>
git lfs pull
```

### Track tokenizer JSON with LFS (contributors)

```bash
git lfs track "*.json"
git add .gitattributes
git commit -m "Track tokenizer JSON with Git LFS"
```
