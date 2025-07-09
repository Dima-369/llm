use crate::{chat::{ChatMessage, ChatProvider, ChatResponse}, completion::{CompletionProvider, CompletionRequest, CompletionResponse}, embedding::EmbeddingProvider, error::LLMError, secret_store::SecretStore, stt::SpeechToTextProvider, tts::TextToSpeechProvider, LLMProvider, ToolCall, models::ModelsProvider,};
use async_trait::async_trait;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use tokio::sync::Mutex;

const ACTIVATION_ENDPOINT: &str = "https://www.codegeneration.ai/activate-v2";
const CHAT_COMPLETIONS_ENDPOINT: &str = "https://api.together.xyz/v1/chat/completions";
const DEFAULT_MODEL: &str = "deepseek-ai/DeepSeek-V3";

#[derive(Debug, Clone)]
pub struct Together {
    api_key: Arc<Mutex<String>>,
    client: reqwest::Client,
}

impl LLMProvider for Together {}

impl Together {
    pub fn new(proxy_url: Option<String>, _secrets: &SecretStore) -> Self {
        let mut builder = reqwest::Client::builder();

        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(&proxy_url).expect("Failed to create proxy");
            builder = builder.proxy(proxy).danger_accept_invalid_certs(true);
        }

        let client = builder
            .build()
            .expect("Failed to build client");

        Self {
            api_key: Arc::new(Mutex::new(String::new())),
            client,
        }
    }

    async fn get_activation_key(&self) -> Result<String, LLMError> {
        let mut api_key_guard = self.api_key.lock().await;
        if !api_key_guard.is_empty() {
            return Ok(api_key_guard.clone());
        }

        let response = self
            .client
            .get(ACTIVATION_ENDPOINT)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| LLMError::Generic(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LLMError::Generic(e.to_string()))?;

        if !status.is_success() {
            return Err(LLMError::Generic(format!(
                "Failed to get activation key: Status {} - {}",
                status, body
            )));
        }

        let json: Value = serde_json::from_str(&body).map_err(|e| {
            LLMError::Generic(format!("Failed to parse activation key response: {}", e))
        })?;
        let key = json["openAIParams"]["apiKey"]
            .as_str()
            .ok_or_else(|| LLMError::Generic("API key not found in response".to_string()))?
            .to_string();

        *api_key_guard = key.clone();
        Ok(key)
    }
}

#[async_trait]
impl CompletionProvider for Together {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let api_key = self.get_activation_key().await?;
        let body = RequestBody::from(request);

        let res = self
            .client
            .post(CHAT_COMPLETIONS_ENDPOINT)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::Generic(e.to_string()))?;

        let status = res.status();
        if !status.is_success() {
            let res_body = res
                .text()
                .await
                .map_err(|e| LLMError::Generic(e.to_string()))?;
            return Err(LLMError::Generic(format!(
                "API call failed with status: {}, body: {}",
                status, res_body
            )));
        }

        let response_body: ResponseBody = res
            .json()
            .await
            .map_err(|e| LLMError::Generic(format!("Failed to parse response body: {}", e)))?;

        Ok(CompletionResponse::from(response_body))
    }
}

#[async_trait]
impl ChatProvider for Together {
    async fn chat(&self, _messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "Chat not supported by the Together completion provider".to_string(),
        ))
    }

    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[crate::chat::Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "Chat with tools not supported by the Together completion provider".to_string(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String, LLMError>> + Send>>,
        LLMError,
    > {
        Err(LLMError::ProviderError(
            "Chat streaming not supported by the Together completion provider".to_string(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for Together {
    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported by the Together completion provider".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for Together {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Speech to text not supported by the Together completion provider".to_string(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Together {
    // No implementation needed for text-to-speech if not supported
}

#[async_trait]
impl ModelsProvider for Together {
    // No implementation needed for models if not supported
}

#[derive(Serialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct RequestBody {
    messages: Vec<Message>,
    model: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

impl From<&CompletionRequest> for RequestBody {
    fn from(request: &CompletionRequest) -> Self {
        Self {
            messages: vec![Message {
                role: "user".to_string(),
                content: request.prompt.clone(),
            }],
            model: DEFAULT_MODEL.to_string(),
            stream: false,
            temperature: request.temperature,
            
            max_tokens: request.max_tokens,
        }
    }
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct ResponseBody {
    choices: Vec<Choice>,
}

impl From<ResponseBody> for CompletionResponse {
    fn from(response: ResponseBody) -> Self {
        let text = response
            .choices
            .get(0)
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        Self { text }
    }
}

#[derive(Debug)]
pub struct TogetherChatResponse {
    text: String,
}

impl ChatResponse for TogetherChatResponse {
    fn text(&self) -> Option<String> {
        Some(self.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }
}

impl fmt::Display for TogetherChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}
