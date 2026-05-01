use std::sync::LazyLock;

use minijinja::{Environment, context};
use minijinja_contrib::pycompat;
use serde::Serialize;
use serde_json::Value;
use tokenizers::{InputSequence, Tokenizer};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to tokenize input: {0}")]
    Tokenization(#[from] tokenizers::tokenizer::Error),
    #[error("Failed to run task on thread pool")]
    ThreadPool(#[from] async_threadpool::Error),
    #[error("Failed to render chat template: {0}")]
    Template(#[from] minijinja::Error),
}

#[cfg(not(feature = "qwen-2.5-0.5b-instruct"))]
compile_error!("feature `qwen-2.5-0.5b-instruct` must be enabled");

const QWEN_TOKENIZER_BYTES: &[u8] = include_bytes!("../qwen2_5_0_5b_instruct.json");
static QWEN_CHAT_TEMPLATE: &str = include_str!("../qwen2_5_0_5b_instruct_chat_template.jinja");

static QWEN_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_bytes(QWEN_TOKENIZER_BYTES).unwrap());

/// A message for tokenization with chat template applied
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    fn new(role: &'static str, content: String) -> Self {
        Self {
            role: role.into(),
            content,
        }
    }

    pub fn system(content: String) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: String) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: String) -> Self {
        Self::new("assistant", content)
    }

    pub fn tool(content: String) -> Self {
        Self::new("tool", content)
    }
}

/// Tokenize raw text per Qwen2.5-0.5B-Instruct (without chat template).
pub async fn qwen<'a, E: Into<InputSequence<'a>> + Send + 'static>(
    input: E,
) -> Result<usize, Error> {
    async_threadpool::run(|| Ok(QWEN_TOKENIZER.encode(input.into(), false)?.len())).await?
}

/// Tokenize messages with the Qwen2.5-0.5B-Instruct chat template.
///
/// This uses the Hugging Face tokenizer chat template consumed by SGLang for
/// `Qwen/Qwen2.5-0.5B-Instruct`, with `add_generation_prompt` enabled.
pub async fn qwen_chat(messages: Vec<ChatMessage>) -> Result<usize, Error> {
    async_threadpool::run(move || {
        let mut env = Environment::new();
        // add support for jinja/python methods like strip
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_template("chat", QWEN_CHAT_TEMPLATE)?;
        let tmpl = env.get_template("chat")?;
        let tools: Vec<Value> = Vec::new();

        let formatted = tmpl.render(context! {
            messages => messages,
            add_generation_prompt => true,
            tools => tools,
        })?;

        Ok(QWEN_TOKENIZER.encode(formatted, false)?.len())
    })
    .await?
}

/// Deprecated compatibility alias for [`qwen`].
#[deprecated(note = "use qwen; this crate now embeds Qwen2.5-0.5B-Instruct")]
pub async fn glm<'a, E: Into<InputSequence<'a>> + Send + 'static>(
    input: E,
) -> Result<usize, Error> {
    qwen(input).await
}

/// Deprecated compatibility alias for [`qwen_chat`].
#[deprecated(note = "use qwen_chat; this crate now embeds Qwen2.5-0.5B-Instruct")]
pub async fn glm_chat(messages: Vec<ChatMessage>) -> Result<usize, Error> {
    qwen_chat(messages).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization() {
        LazyLock::force(&QWEN_TOKENIZER);
    }

    #[tokio::test]
    async fn test_qwen() {
        let input = "Hello, world!";
        let result = qwen(input).await.unwrap();
        assert_eq!(result, 4);
    }

    #[tokio::test]
    async fn test_qwen_chat() {
        let messages = vec![ChatMessage::user("Hello, world!".to_owned())];
        let result = qwen_chat(messages).await.unwrap();
        assert_eq!(result, 33);
    }

    #[tokio::test]
    async fn test_qwen_chat_vs_raw() {
        let content = "Hello, world!";
        let raw_tokens = qwen(content).await.unwrap();
        let chat_tokens = qwen_chat(vec![ChatMessage::user(content.into())])
            .await
            .unwrap();

        println!("Raw tokens: {raw_tokens}, Chat tokens: {chat_tokens}");
        assert!(chat_tokens > raw_tokens);
    }

    #[tokio::test]
    async fn test_qwen_chat_with_assistant() {
        let messages = vec![
            ChatMessage::user("Hello!".to_owned()),
            ChatMessage::assistant("Hi there!".to_owned()),
        ];
        let result = qwen_chat(messages).await.unwrap();
        assert!(result > 0);
    }
}
