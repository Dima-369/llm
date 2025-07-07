//! Copilot API client implementation.
//!
//! This module provides integration with GitHub Copilot's chat functionality.
//! It handles the complete authentication flow, including the OAuth2 device flow
//! for obtaining a GitHub token, and subsequent fetching of a short-lived Copilot token.
//! Tokens are cached locally in `~/.llm/`.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole},
    chat::{Tool, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider, ToolCall,
};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::time::sleep;

// --- Constants ---
const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const EDITOR_VERSION: &str = "vscode/1.85.1";
const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.11.1";

// --- API Structs ---

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CopilotToken {
    pub token: String,
    pub expires_at: i64,
}

#[derive(Deserialize)]
pub struct GithubTokenResponse {
    pub access_token: String,
}

#[derive(Serialize)]
struct CopilotChatRequest<'a> {
    model: &'a str,
    messages: Vec<CopilotChatMessage<'a>>,
    stream: bool,
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

#[derive(Serialize)]
struct CopilotChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize, Debug)]
struct CopilotChatResponse {
    choices: Vec<CopilotChatChoice>,
}

#[derive(Deserialize, Debug)]
struct CopilotChatChoice {
    message: CopilotChatMsg,
}

#[derive(Deserialize, Debug)]
struct CopilotChatMsg {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

impl std::fmt::Display for CopilotChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

impl ChatResponse for CopilotChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }
}

// --- Provider Struct ---

/// Client for interacting with GitHub Copilot's API.
pub struct Copilot {
    client: Client,
    github_token: String,
    copilot_token: Arc<RwLock<Option<CopilotToken>>>,
    model: String,
    temperature: Option<f32>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    token_directory: PathBuf,
}

impl Copilot {
    fn llm_dir(&self) -> Result<PathBuf, LLMError> {
        fs::create_dir_all(&self.token_directory)
            .map_err(|e| LLMError::Generic(format!("Failed to create token directory: {e}")))?;
        Ok(self.token_directory.clone())
    }

    fn github_token_file(&self) -> Result<PathBuf, LLMError> {
        Ok(self.llm_dir()?.join("copilot_github_token.json"))
    }

    fn copilot_token_file(&self) -> Result<PathBuf, LLMError> {
        Ok(self.llm_dir()?.join("copilot_token.json"))
    }

    /// Creates a new Copilot client.
    /// If a GitHub token is not provided, it will attempt to load one from the cache,
    /// or initiate an interactive device authentication flow.
    pub fn new(
        github_token: Option<String>,
        proxy_url: Option<String>,
        model: Option<String>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        token_directory: PathBuf,
    ) -> Result<Self, LLMError> {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(Duration::from_secs(sec));
        }
        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(&proxy_url).expect("Failed to create proxy");
            builder = builder.proxy(proxy).danger_accept_invalid_certs(true);
        }
        let client = builder
            .build()
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let dummy_copilot = Self {
            client: client.clone(),
            github_token: String::new(),                // Placeholder
            copilot_token: Arc::new(RwLock::new(None)), // Placeholder
            model: String::new(),                       // Placeholder
            temperature: None,                          // Placeholder
            tools: None,                                // Placeholder
            tool_choice: None,                          // Placeholder
            token_directory: token_directory.clone(),
        };

        let final_github_token = match github_token {
            Some(token) => {
                log::debug!("Using provided GitHub token.");
                token
            }
            None => {
                log::debug!("No GitHub token provided, checking cache.");
                match dummy_copilot.load_github_token() {
                    Ok(token) => {
                        log::debug!("Loaded GitHub token from cache.");
                        token
                    }
                    Err(_) => {
                        log::info!("No cached GitHub token. Starting interactive authentication.");
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current()
                                .block_on(dummy_copilot.interactive_github_auth(&client))
                        })?
                    }
                }
            }
        };

        let cached_copilot_token = dummy_copilot.load_copilot_token().ok();

        Ok(Self {
            client,
            github_token: final_github_token,
            copilot_token: Arc::new(RwLock::new(cached_copilot_token)),
            model: model.unwrap_or("copilot-chat".to_string()),
            temperature,
            tools,
            tool_choice,
            token_directory,
        })
    }

    /// Handles the interactive device flow to get a GitHub token.
    async fn interactive_github_auth(&self, client: &Client) -> Result<String, LLMError> {
        let device_code_response = client
            .post("https://github.com/login/device/code")
            .header("Accept", "application/json")
            .form(&[("client_id", GITHUB_CLIENT_ID)])
            .send()
            .await?;

        let device_code_json: serde_json::Value = device_code_response.json().await?;
        let device_code = device_code_json["device_code"]
            .as_str()
            .ok_or_else(|| LLMError::Generic("No device_code in response".into()))?;
        let user_code = device_code_json["user_code"]
            .as_str()
            .ok_or_else(|| LLMError::Generic("No user_code in response".into()))?;
        let verification_uri = device_code_json["verification_uri"]
            .as_str()
            .ok_or_else(|| LLMError::Generic("No verification_uri in response".into()))?;

        println!("\nFirst time use requires authentication with GitHub.");
        println!("Opened {verification_uri} in your browser.");
        println!("Please enter the following code: {user_code}");

        // Attempt to open in browser, but don't fail if it doesn't work.
        let _ = open::that(verification_uri);

        let max_attempts = 30; // 5 minutes
        for attempt in 0..max_attempts {
            sleep(Duration::from_secs(10)).await;
            let response = client
                .post("https://github.com/login/oauth/access_token")
                .header("Accept", "application/json")
                .form(&[
                    ("client_id", GITHUB_CLIENT_ID),
                    ("device_code", device_code),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ])
                .send()
                .await?;

            let status = response.status();
            let response_text = response.text().await?;

            if status.is_success() {
                if let Ok(token_res) = serde_json::from_str::<GithubTokenResponse>(&response_text) {
                    if !token_res.access_token.is_empty() {
                        println!("Successfully authenticated with GitHub.");
                        self.save_github_token(&token_res.access_token)?;
                        return Ok(token_res.access_token);
                    }
                }
            }

            if !response_text.contains("authorization_pending") {
                return Err(LLMError::AuthError(format!(
                    "GitHub token request failed with status {status}: {response_text}"
                )));
            }

            if attempt == max_attempts - 1 {
                return Err(LLMError::AuthError(
                    "GitHub authentication timed out.".into(),
                ));
            }
        }

        Err(LLMError::AuthError(
            "GitHub authentication timed out.".into(),
        ))
    }

    /// Gets a fresh Copilot token, refreshing if necessary.
    async fn get_refreshed_copilot_token(&self) -> Result<CopilotToken, LLMError> {
        let should_refresh = {
            let read_lock = self.copilot_token.read().unwrap();
            match &*read_lock {
                Some(token) => token.expires_at < Utc::now().timestamp() + 300, // refresh if expires in 5 mins
                None => true,
            }
        };

        if should_refresh {
            log::debug!("Copilot token expired or missing, fetching a new one.");
            let new_token = self.fetch_copilot_token_from_api().await?;
            let mut write_lock = self.copilot_token.write().unwrap();
            *write_lock = Some(new_token.clone());
            self.save_copilot_token(&new_token)?;
            Ok(new_token)
        } else {
            log::debug!("Using cached Copilot token.");
            let read_lock = self.copilot_token.read().unwrap();
            Ok(read_lock.clone().unwrap()) // Safe to unwrap due to check above
        }
    }

    /// Fetches a new Copilot token from the GitHub API.
    async fn fetch_copilot_token_from_api(&self) -> Result<CopilotToken, LLMError> {
        let response = self
            .client
            .get("https://api.github.com/copilot_internal/v2/token")
            .header("authorization", format!("token {}", self.github_token))
            .header("accept", "application/json")
            .header("editor-version", EDITOR_VERSION)
            .header("editor-plugin-version", EDITOR_PLUGIN_VERSION)
            .header("user-agent", EDITOR_PLUGIN_VERSION)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::AuthError(format!(
                "Copilot token request failed with status {status}: {error_text}"
            )));
        }

        response
            .json::<CopilotToken>()
            .await
            .map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: "Failed to parse CopilotToken".into(),
            })
    }

    // --- Token Caching ---
    fn load_github_token(&self) -> Result<String, LLMError> {
        let content = fs::read_to_string(self.github_token_file()?)
            .map_err(|e| LLMError::Generic(e.to_string()))?;
        Ok(content)
    }

    fn save_github_token(&self, token: &str) -> Result<(), LLMError> {
        fs::write(self.github_token_file()?, token).map_err(|e| LLMError::Generic(e.to_string()))
    }

    fn load_copilot_token(&self) -> Result<CopilotToken, LLMError> {
        let content = fs::read_to_string(self.copilot_token_file()?)
            .map_err(|e| LLMError::Generic(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| LLMError::JsonError(e.to_string()))
    }

    fn save_copilot_token(&self, token: &CopilotToken) -> Result<(), LLMError> {
        let content =
            serde_json::to_string(token).map_err(|e| LLMError::JsonError(e.to_string()))?;
        fs::write(self.copilot_token_file()?, content).map_err(|e| LLMError::Generic(e.to_string()))
    }
}

#[async_trait]
impl ChatProvider for Copilot {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let fresh_token = self.get_refreshed_copilot_token().await?;

        let copilot_messages: Vec<CopilotChatMessage> = messages
            .iter()
            .map(|m| CopilotChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());

        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let body = CopilotChatRequest {
            model: &self.model,
            messages: copilot_messages,
            stream: false,
            temperature: self.temperature,
            tools: request_tools,
            tool_choice: request_tool_choice,
        };

        let response = self
            .client
            .post("https://api.githubcopilot.com/chat/completions")
            .header("authorization", format!("Bearer {}", fresh_token.token))
            .header("accept", "*/*")
            .header("editor-version", EDITOR_VERSION)
            .header("editor-plugin-version", EDITOR_PLUGIN_VERSION)
            .header("user-agent", EDITOR_PLUGIN_VERSION)
            .header("copilot-integration-id", "vscode-chat")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::ProviderError(format!(
                "Copilot chat request failed with status {status}: {error_text}"
            )));
        }

        response
            .json::<CopilotChatResponse>()
            .await
            .map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: "Failed to parse CopilotChatResponse".into(),
            })
            .map(|r| Box::new(r) as Box<dyn ChatResponse>)
    }
}

#[async_trait]
impl CompletionProvider for Copilot {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Err(LLMError::ProviderError("Completion not supported".into()))
    }
}

#[async_trait]
impl EmbeddingProvider for Copilot {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError("Embedding not supported".into()))
    }
}

#[async_trait]
impl SpeechToTextProvider for Copilot {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError("STT not supported".into()))
    }
}

#[async_trait]
impl TextToSpeechProvider for Copilot {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError("TTS not supported".into()))
    }
}

#[async_trait]
impl ModelsProvider for Copilot {}
impl LLMProvider for Copilot {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }

    fn relogin_github_copilot(&self) -> Result<(), crate::error::LLMError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.interactive_github_auth(&self.client))
        })?;
        Ok(())
    }
}
