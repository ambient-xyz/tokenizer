# tokenizer

Async token counting utilities for Ambient's GLM serving path.

This crate is currently pinned to the GLM-5.1 FP8 serving assets:

- Model: `zai-org/GLM-5.1-FP8`
- Model revision: `f396cf805182f4ca10fa675e1a99815b3ca384db`
- Tokenizer: `glm.json`
- Tokenizer SHA256: `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`
- Chat template: `glm_5_1_chat_template.jinja`
- Chat template SHA256: `7e11a0b0081fb7ebb2280f8ac320a2c3816201a792241ee77c9f9bdf26d779cf`
- Max model length: `202752`

The tokenizer and chat template must match the SGLang model snapshot used by
miners. Cache-aware routing depends on that identity; changing these assets
changes token counts and token-block hashes.

The bundled template is semantically equivalent to the upstream GLM-5.1
`chat_template.jinja`, with one compatibility edit for MiniJinja:
`m.content.0.type` is represented as `m.content[0].type`.

## API

- `glm(...)`: count tokens for raw text without applying a chat template.
- `glm_token_ids(...)`: return token ids for raw text.
- `glm_chat(...)`: count tokens after rendering the GLM-5.1 chat template with
  default options.
- `glm_chat_with_options(...)`: count chat tokens with explicit template flags
  such as `enable_thinking`.
- `glm_chat_token_ids(...)`: return token ids for chat messages with default
  template options.
- `glm_chat_token_ids_with_options(...)`: return token ids for chat messages
  with explicit template options.
- `glm_tokenizer_identity()`: return model, tokenizer, and template identity
  metadata for downstream cache-key scoping.

## Quick Start

```rust
use tokenizer::{glm_chat, glm_tokenizer_identity, ChatMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let identity = glm_tokenizer_identity();
    println!("tokenizer: {:?}", identity);

    let messages = vec![
        ChatMessage::system("You are a helpful assistant.".to_string()),
        ChatMessage::user("Hello, world!".to_string()),
    ];

    let n = glm_chat(messages).await?;
    println!("chat tokens: {n}");
    Ok(())
}
```

## Template Options

GLM-5.1's chat template has flags that affect serialized prompt tokens. Use
`ChatTemplateOptions` when a serving path sets those flags explicitly.

```rust
use tokenizer::{glm_chat_with_options, ChatMessage, ChatTemplateOptions};

let tokens = glm_chat_with_options(
    vec![ChatMessage::user("Hello".to_string())],
    ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    },
).await?;
```

Leaving an option as `None` omits it from the template context, preserving the
upstream template's default behavior.

## Determinism

`glm_chat` output is sensitive to:

- model revision
- tokenizer JSON
- chat template
- `add_generation_prompt`
- `enable_thinking`
- `clear_thinking`

Downstream services should store `glm_tokenizer_identity()` alongside cache-key
or accounting metadata.

## Testing

```bash
cargo test
```

The tests initialize the bundled tokenizer, validate identity metadata, compare
raw and chat token counts, and ensure token-id APIs match count APIs.
