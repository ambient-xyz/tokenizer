use std::sync::LazyLock;

use minijinja::Environment;
use minijinja_contrib::pycompat;
use serde::Serialize;
use tokenizers::{InputSequence, Tokenizer};

pub const GLM_MODEL_ID: &str = "zai-org/GLM-5.1-FP8";
pub const GLM_MODEL_REVISION: &str = "f396cf805182f4ca10fa675e1a99815b3ca384db";
pub const GLM_TOKENIZER_FILENAME: &str = "glm.json";
pub const GLM_TOKENIZER_SHA256: &str =
    "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d";
pub const GLM_CHAT_TEMPLATE_FILENAME: &str = "glm_5_1_chat_template.jinja";
pub const GLM_CHAT_TEMPLATE_SHA256: &str =
    "7e11a0b0081fb7ebb2280f8ac320a2c3816201a792241ee77c9f9bdf26d779cf";
pub const GLM_MODEL_MAX_LENGTH: usize = 202_752;

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

static GLM_CHAT_TEMPLATE: &str = include_str!("../glm_5_1_chat_template.jinja");

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct TokenizerIdentity {
    pub model_id: &'static str,
    pub model_revision: &'static str,
    pub tokenizer_filename: &'static str,
    pub tokenizer_sha256: &'static str,
    pub chat_template_filename: &'static str,
    pub chat_template_sha256: &'static str,
    pub model_max_length: usize,
}

pub fn glm_tokenizer_identity() -> TokenizerIdentity {
    TokenizerIdentity {
        model_id: GLM_MODEL_ID,
        model_revision: GLM_MODEL_REVISION,
        tokenizer_filename: GLM_TOKENIZER_FILENAME,
        tokenizer_sha256: GLM_TOKENIZER_SHA256,
        chat_template_filename: GLM_CHAT_TEMPLATE_FILENAME,
        chat_template_sha256: GLM_CHAT_TEMPLATE_SHA256,
        model_max_length: GLM_MODEL_MAX_LENGTH,
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub clear_thinking: Option<bool>,
}

#[derive(Serialize)]
struct ChatTemplateContext {
    messages: Vec<ChatMessage>,
    add_generation_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    clear_thinking: Option<bool>,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: None,
            clear_thinking: None,
        }
    }
}

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
    Ok(glm_token_ids(input).await?.len())
}

/// Tokenize raw text per GLM and return token ids.
pub async fn glm_token_ids<'a, E: Into<InputSequence<'a>> + Send + 'static>(
    input: E,
) -> Result<Vec<u32>, Error> {
    async_threadpool::run(|| {
        Ok(GLM_TOKENIZER
            .encode(input.into(), false)?
            .get_ids()
            .to_vec())
    })
    .await?
}

/// Tokenize messages (with GLM chat template)
pub async fn glm_chat(messages: Vec<ChatMessage>) -> Result<usize, Error> {
    glm_chat_with_options(messages, ChatTemplateOptions::default()).await
}

/// Tokenize messages with explicit GLM chat template options.
pub async fn glm_chat_with_options(
    messages: Vec<ChatMessage>,
    options: ChatTemplateOptions,
) -> Result<usize, Error> {
    Ok(glm_chat_token_ids_with_options(messages, options)
        .await?
        .len())
}

/// Tokenize messages with the default GLM chat template options and return token ids.
pub async fn glm_chat_token_ids(messages: Vec<ChatMessage>) -> Result<Vec<u32>, Error> {
    glm_chat_token_ids_with_options(messages, ChatTemplateOptions::default()).await
}

/// Tokenize messages with explicit GLM chat template options and return token ids.
pub async fn glm_chat_token_ids_with_options(
    messages: Vec<ChatMessage>,
    options: ChatTemplateOptions,
) -> Result<Vec<u32>, Error> {
    async_threadpool::run(move || {
        let mut env = Environment::new();
        // add support for jinja/python methods like strip
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_template("chat", GLM_CHAT_TEMPLATE)?;
        let tmpl = env.get_template("chat")?;

        let formatted = tmpl.render(ChatTemplateContext {
            messages,
            add_generation_prompt: options.add_generation_prompt,
            enable_thinking: options.enable_thinking,
            clear_thinking: options.clear_thinking,
        })?;

        Ok(GLM_TOKENIZER.encode(formatted, false)?.get_ids().to_vec())
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

    #[test]
    fn identity_tracks_glm_5_1_assets() {
        let identity = glm_tokenizer_identity();
        assert_eq!(identity.model_id, "zai-org/GLM-5.1-FP8");
        assert_eq!(
            identity.model_revision,
            "f396cf805182f4ca10fa675e1a99815b3ca384db"
        );
        assert_eq!(
            identity.chat_template_filename,
            "glm_5_1_chat_template.jinja"
        );
        assert_eq!(identity.model_max_length, 202_752);
    }

    #[tokio::test]
    async fn test_glm() {
        let input = "Hello, world!";
        let result = glm(input).await.unwrap();
        assert!(result > 0);
    }

    #[tokio::test]
    async fn test_glm_token_ids_match_count() {
        let input = "Hello, world!";
        let count = glm(input).await.unwrap();
        let ids = glm_token_ids(input).await.unwrap();
        assert_eq!(ids.len(), count);
    }

    #[tokio::test]
    async fn test_glm_chat() {
        let messages = vec![ChatMessage::user("Hello, world!".to_owned())];
        let result = glm_chat(messages).await.unwrap();
        // Should be more than raw tokenization due to template tokens
        assert!(result > 4);
    }

    #[tokio::test]
    async fn test_glm_chat_token_ids_match_count() {
        let messages = vec![ChatMessage::user("Hello, world!".to_owned())];
        let count = glm_chat(messages.clone()).await.unwrap();
        let ids = glm_chat_token_ids(messages).await.unwrap();
        assert_eq!(ids.len(), count);
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
    async fn test_glm_chat_with_template_options() {
        let messages = vec![ChatMessage::user("Hello!".to_owned())];
        let result = glm_chat_with_options(
            messages,
            ChatTemplateOptions {
                enable_thinking: Some(false),
                ..ChatTemplateOptions::default()
            },
        )
        .await
        .unwrap();
        assert!(result > 0);
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
