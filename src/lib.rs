use std::sync::LazyLock;

use minijinja::{Environment, context};
use minijinja_contrib::pycompat;
use serde::Serialize;
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

static GLM_TOKENIZER: LazyLock<Tokenizer> =
    LazyLock::new(|| Tokenizer::from_bytes(include_bytes!("../glm.json")).unwrap());

static GLM_CHAT_TEMPLATE: &str = include_str!("../glm_4_6_chat_template.jinja");

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

/// Tokenize raw text per GLM (without chat template)
pub async fn glm<'a, E: Into<InputSequence<'a>> + Send + 'static>(
    input: E,
) -> Result<usize, Error> {
    async_threadpool::run(|| Ok(GLM_TOKENIZER.encode(input.into(), false)?.len())).await?
}

/// Tokenize messages (with GLM chat template)
pub async fn glm_chat(messages: Vec<ChatMessage>) -> Result<usize, Error> {
    async_threadpool::run(move || {
        let mut env = Environment::new();
        // add support for jinja/python methods like strip
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_template("chat", GLM_CHAT_TEMPLATE)?;
        let tmpl = env.get_template("chat")?;

        let formatted = tmpl.render(context! {
            messages => messages,
            add_generation_prompt => true,
        })?;

        Ok(GLM_TOKENIZER.encode(formatted, false)?.len())
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization() {
        LazyLock::force(&GLM_TOKENIZER);
    }

    #[tokio::test]
    async fn test_glm() {
        let input = "Hello, world!";
        let result = glm(input).await.unwrap();
        assert_eq!(result, 4);
    }

    #[tokio::test]
    async fn test_glm_chat() {
        let messages = vec![ChatMessage::user("Hello, world!".to_owned())];
        let result = glm_chat(messages).await.unwrap();
        // Should be more than raw tokenization due to template tokens
        assert!(result > 4);
    }

    #[tokio::test]
    async fn test_glm_chat_vs_raw() {
        let content = "Hello, world!";
        let raw_tokens = glm(content).await.unwrap();
        let chat_tokens = glm_chat(vec![ChatMessage::user(content.into())])
            .await
            .unwrap();

        println!("Raw tokens: {raw_tokens}, Chat tokens: {chat_tokens}");
        assert!(chat_tokens > raw_tokens);
    }

    #[tokio::test]
    async fn test_glm_chat_with_assistant() {
        let messages = vec![
            ChatMessage::user("Hello!".to_owned()),
            ChatMessage::assistant("Hi there!".to_owned()),
        ];
        let result = glm_chat(messages).await.unwrap();
        assert!(result > 0);
    }
}
