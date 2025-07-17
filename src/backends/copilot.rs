//! Copilot API client implementation.
//!
//! This module provides integration with GitHub Copilot's chat functionality.
//! It handles the complete authentication flow, including the OAuth2 device flow
//! for obtaining a GitHub token, and subsequent fetching of a short-lived Copilot token.
//! Tokens are cached locally in `~/.llm/`.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType},
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
use chrono::{DateTime, Local, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::time::sleep;

// Helper functions for clipboard and notifications
fn copy_to_clipboard(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut clipboard = arboard::Clipboard::new()?;
    clipboard.set_text(text)?;
    Ok(())
}

fn show_hammerspoon_alert(s: &str) {
    use std::process::{Command, Stdio};
    let _ = Command::new("hs")
        .arg("-c")
        .arg("hs.alert.show([===[".to_owned() +
            s +
            "]===],{ textStyle = { paragraphStyle = { alignment = \"center\" } } }, hs.screen.mainScreen(), 2)")
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .spawn();
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
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
        // Look for content in any choice, prioritizing the first one with content
        for choice in &self.choices {
            if let Some(content) = &choice.message.content {
                if !content.trim().is_empty() {
                    return Some(content.clone());
                }
            }
        }
        None
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        // Look for tool calls in any choice
        for choice in &self.choices {
            if let Some(tool_calls) = &choice.message.tool_calls {
                if !tool_calls.is_empty() {
                    return Some(tool_calls.clone());
                }
            }
        }
        None
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
    system: Option<String>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        github_token: Option<String>,
        proxy_url: Option<String>,
        model: Option<String>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
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
            system: None,                               // Placeholder
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
            system,
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

        // Copy the user code to clipboard
        if let Err(e) = copy_to_clipboard(user_code) {
            eprintln!("Warning: Failed to copy code to clipboard: {}", e);
        }

        // Show Hammerspoon notification
        show_hammerspoon_alert(&format!("GitHub Copilot Auth\n\nCode copied to clipboard:\n{}", user_code));

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

    /// Fetches GitHub Copilot usage information from the API.
    async fn fetch_copilot_usage(&self) -> Result<crate::CopilotUsageInfo, LLMError> {
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

        let status = response.status();
        let response_text = response.text().await?;

        if !status.is_success() {
            return Err(LLMError::AuthError(format!(
                "Copilot usage request failed with status {status}: {response_text}"
            )));
        }

        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| LLMError::JsonError(e.to_string()))?;

        let reset_date = response_json["limited_user_reset_date"]
            .as_i64()
            .unwrap_or_default();
        let reset_date_local: DateTime<Local> = DateTime::from(
            Utc.timestamp_opt(reset_date, 0)
                .single()
                .unwrap_or_default(),
        );

        let now = Local::now();
        let duration = reset_date_local.signed_duration_since(now);
        let hours_remaining = duration.num_hours();

        let chat_left = response_json["limited_user_quotas"]["chat"]
            .as_i64()
            .unwrap_or_default();
        let completions_left = response_json["limited_user_quotas"]["completions"]
            .as_i64()
            .unwrap_or_default();

        Ok(crate::CopilotUsageInfo {
            chat_messages_left_per_month: chat_left,
            completions_left_per_month: completions_left,
            reset_date: reset_date_local.format("%Y-%m-%d %H:%M:%S %Z").to_string(),
            time_remaining: format!(
                "{} days {} hours",
                hours_remaining / 24,
                hours_remaining % 24
            ),
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

        let mut copilot_messages: Vec<CopilotChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    copilot_messages.push(CopilotChatMessage {
                        role: "tool",
                        content: &result.function.arguments,
                        tool_calls: None,
                        tool_call_id: Some(result.id.clone()),
                    });
                }
            } else {
                copilot_messages.push(chat_message_to_copilot_message(msg))
            }
        }

        // Insert system message at the beginning if present
        if let Some(system) = &self.system {
            copilot_messages.insert(
                0,
                CopilotChatMessage {
                    role: "system",
                    content: system,
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

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

// Create an owned CopilotChatMessage that doesn't borrow from any temporary variables
fn chat_message_to_copilot_message(chat_msg: &ChatMessage) -> CopilotChatMessage<'static> {
    CopilotChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        },
        content: match &chat_msg.message_type {
            MessageType::Text => {
                // Leak the string to get a 'static reference
                Box::leak(chat_msg.content.clone().into_boxed_str())
            }
            MessageType::ToolUse(_) => "",
            MessageType::ToolResult(_) => "",
            _ => {
                // For other message types, use the content field
                Box::leak(chat_msg.content.clone().into_boxed_str())
            }
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => Some(calls.clone()),
            _ => None,
        },
        tool_call_id: None,
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
        
        // Clear the cached Copilot token so a fresh one will be fetched for the new GitHub account
        {
            let mut write_lock = self.copilot_token.write().unwrap();
            *write_lock = None;
        }
        
        // Also remove the cached token file
        if let Ok(token_file) = self.copilot_token_file() {
            let _ = std::fs::remove_file(token_file);
        }
        
        Ok(())
    }

    fn get_github_copilot_usage(&self) -> Result<crate::CopilotUsageInfo, crate::error::LLMError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.fetch_copilot_usage())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copilot_response_parsing_with_multiple_choices() {
        // Test case based on the actual GitHub Copilot response structure
        // where tool calls are in the second choice, not the first
        let response_json = r#"{
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "I'll search for and open any README files in the workspace.",
                        "role": "assistant"
                    }
                },
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{\"command\":\"find . -iname \\\"readme*\\\"\"}",
                                    "name": "bash"
                                },
                                "id": "tooluse_HTIc-tQBS8CEYLO6OdSPPA",
                                "type": "function"
                            }
                        ]
                    }
                }
            ],
            "created": 1751913371,
            "id": "54f2dd85-f04b-45f4-86b1-e74b5e4a2615",
            "usage": {
                "completion_tokens": 73,
                "prompt_tokens": 855,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "total_tokens": 928
            },
            "model": "Claude Sonnet 3.5"
        }"#;

        let response: CopilotChatResponse = serde_json::from_str(response_json).unwrap();

        // Test that we can extract the content from the first choice
        assert_eq!(
            response.text(),
            Some("I'll search for and open any README files in the workspace.".to_string())
        );

        // Test that we can extract the tool calls from the second choice
        let tool_calls = response.tool_calls().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tooluse_HTIc-tQBS8CEYLO6OdSPPA");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "bash");
        assert_eq!(
            tool_calls[0].function.arguments,
            r#"{"command":"find . -iname \"readme*\""}"#
        );
    }

    #[test]
    fn test_copilot_response_parsing_with_single_choice() {
        // Test case for a normal response with content only
        let response_json = r#"{
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you today?",
                        "role": "assistant"
                    }
                }
            ]
        }"#;

        let response: CopilotChatResponse = serde_json::from_str(response_json).unwrap();

        assert_eq!(
            response.text(),
            Some("Hello! How can I help you today?".to_string())
        );
        assert!(response.tool_calls().is_none());
    }

    #[test]
    fn test_copilot_response_parsing_with_empty_content() {
        // Test case where content is empty or whitespace
        let response_json = r#"{
            "choices": [
                {
                    "message": {
                        "content": "   ",
                        "role": "assistant"
                    }
                },
                {
                    "message": {
                        "content": "Actual content here",
                        "role": "assistant"
                    }
                }
            ]
        }"#;

        let response: CopilotChatResponse = serde_json::from_str(response_json).unwrap();

        // Should skip the empty/whitespace content and return the actual content
        assert_eq!(response.text(), Some("Actual content here".to_string()));
    }

    #[test]
    fn test_copilot_system_prompt_insertion() {
        use crate::chat::{ChatMessage, ChatRole};
        use tempfile::tempdir;

        // Create a temporary directory for tokens
        let temp_dir = tempdir().unwrap();
        let token_directory = temp_dir.path().to_path_buf();

        // Create a Copilot instance with a system prompt
        let copilot = Copilot {
            client: reqwest::Client::new(),
            github_token: "test_token".to_string(),
            copilot_token: Arc::new(RwLock::new(None)),
            model: "copilot-chat".to_string(),
            temperature: Some(0.7),
            system: Some("You are a helpful assistant.".to_string()),
            tools: None,
            tool_choice: None,
            token_directory,
        };

        // Create test messages
        let messages = [
            ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
                message_type: Default::default(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "Hi there!".to_string(),
                message_type: Default::default(),
            },
        ];

        // Convert messages to Copilot format (simulating the chat method logic)
        let mut copilot_messages: Vec<CopilotChatMessage> = messages
            .iter()
            .map(|m| CopilotChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();

        // Insert system message at the beginning if present (simulating the chat method logic)
        if let Some(system) = &copilot.system {
            copilot_messages.insert(
                0,
                CopilotChatMessage {
                    role: "system",
                    content: system,
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        // Verify that the system message was inserted at the beginning
        assert_eq!(copilot_messages.len(), 3);
        assert_eq!(copilot_messages[0].role, "system");
        assert_eq!(copilot_messages[0].content, "You are a helpful assistant.");
        assert_eq!(copilot_messages[1].role, "user");
        assert_eq!(copilot_messages[1].content, "Hello");
        assert_eq!(copilot_messages[2].role, "assistant");
        assert_eq!(copilot_messages[2].content, "Hi there!");
    }
}
