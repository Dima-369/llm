//! AgentRouter API client implementation for chat functionality.
//!
//! This module provides integration with AgentRouter's LLM models through their API.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, Tool, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider, ToolCall,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Client for interacting with AgentRouter's API.
pub struct AgentRouter {
    pub api_key: Option<String>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    client: Client,
}

#[derive(Serialize)]
struct AgentRouterChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct AgentRouterChatRequest<'a> {
    model: &'a str,
    messages: Vec<AgentRouterChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

#[derive(Deserialize, Debug)]
struct AgentRouterChatResponse {
    choices: Vec<AgentRouterChatChoice>,
}

#[derive(Deserialize, Debug)]
struct AgentRouterChatChoice {
    message: AgentRouterChatMsg,
}

#[derive(Deserialize, Debug)]
struct AgentRouterChatMsg {
    content: String,
}

impl std::fmt::Display for AgentRouterChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

impl ChatResponse for AgentRouterChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| {
            if c.message.content.is_empty() {
                None
            } else {
                Some(c.message.content.clone())
            }
        })
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        todo!()
    }
}

#[allow(clippy::too_many_arguments)]
impl AgentRouter {
    /// Creates a new AgentRouter client with the specified configuration.
    pub fn new(
        api_key: Option<impl Into<String>>,
        proxy_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(&proxy_url).expect("Failed to create proxy");
            builder = builder.proxy(proxy).danger_accept_invalid_certs(true);
        }
        Self {
            api_key: api_key.map(Into::into),
            model: model.unwrap_or("glm-4.6".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for AgentRouter {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut agentrouter_msgs: Vec<AgentRouterChatMessage> = messages
            .iter()
            .map(|m| AgentRouterChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            agentrouter_msgs.insert(
                0,
                AgentRouterChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = AgentRouterChatRequest {
            model: &self.model,
            messages: agentrouter_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("AgentRouter request payload: {}", json);
            }
        }

        let mut request = self
            .client
            .post("https://agentrouter.org/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("User-Agent", "RooCode/3.34.8")
            .header("X-Title", "Roo Code")
            .header("HTTP-Referer", "https://github.com/RooVetGit/Roo-Cline")
            .header("X-Stainless-Runtime-Version", "v22.20.0")
            .header("X-Stainless-Runtime", "node")
            .header("X-Stainless-Arch", "x64")
            .header("X-Stainless-OS", "Linux")
            .header("X-Stainless-Lang", "js");
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        request = request.json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("AgentRouter HTTP status: {}", resp.status());

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let raw_response = resp.text().await?;
            return Err(LLMError::TooManyRequests(raw_response));
        }

        let json_resp: AgentRouterChatResponse = resp.error_for_status()?.json().await?;

        Ok(Box::new(json_resp))
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut agentrouter_msgs: Vec<AgentRouterChatMessage> = messages
            .iter()
            .map(|m| AgentRouterChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            agentrouter_msgs.insert(
                0,
                AgentRouterChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        // Use the provided tools or fallback to stored tools
        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let body = AgentRouterChatRequest {
            model: &self.model,
            messages: agentrouter_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("AgentRouter request payload: {}", json);
            }
        }

        let mut request = self
            .client
            .post("https://agentrouter.org/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("User-Agent", "RooCode/3.34.8")
            .header("X-Title", "Roo Code")
            .header("HTTP-Referer", "https://github.com/RooVetGit/Roo-Cline")
            .header("X-Stainless-Runtime-Version", "v22.20.0")
            .header("X-Stainless-Runtime", "node")
            .header("X-Stainless-Arch", "x64")
            .header("X-Stainless-OS", "Linux")
            .header("X-Stainless-Lang", "js");
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        request = request.json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("AgentRouter HTTP status: {}", resp.status());

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let raw_response = resp.text().await?;
            return Err(LLMError::TooManyRequests(raw_response));
        }

        let json_resp: AgentRouterChatResponse = resp.error_for_status()?.json().await?;

        Ok(Box::new(json_resp))
    }
}

#[async_trait]
impl CompletionProvider for AgentRouter {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "AgentRouter completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for AgentRouter {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for AgentRouter {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "AgentRouter does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for AgentRouter {}

#[async_trait]
impl ModelsProvider for AgentRouter {}

impl LLMProvider for AgentRouter {}