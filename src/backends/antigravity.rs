//! AntiGravity API client implementation.
//!
//! This module provides integration with Google's AntiGravity service (Cloud Code Assist),
//! which supports Gemini and Claude models.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, Tool, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    FunctionCall, LLMProvider, ToolCall,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

// --- Constants ---
const ANTIGRAVITY_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
const ANTIGRAVITY_CLIENT_SECRET: &str = "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";
const ANTIGRAVITY_REDIRECT_URI: &str = "http://localhost:51121/oauth-callback";
const ANTIGRAVITY_CALLBACK_PORT: u16 = 51121;

const ANTIGRAVITY_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];

const ANTIGRAVITY_SANDBOX_ENDPOINTS: &[&str] =
    &["https://daily-cloudcode-pa.sandbox.googleapis.com"];
const ANTIGRAVITY_STREAM_ENDPOINTS: &[&str] = &[
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
];
const ANTIGRAVITY_LOAD_ENDPOINTS: &[&str] = &[
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
];

const ANTIGRAVITY_CLIENT_METADATA: &str =
    r#"{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}"#;
const ANTIGRAVITY_LOAD_USER_AGENT: &str = "google-api-nodejs-client/9.15.1";
const ANTIGRAVITY_API_CLIENT: &str = "google-cloud-sdk vscode_cloudshelleditor/0.1";
const ANTIGRAVITY_GEMINI_CLI_USER_AGENT: &str = "google-cloud-sdk vscode_cloudshelleditor/0.1";
const ANTIGRAVITY_SANDBOX_USER_AGENT_PREFIX: &str = "antigravity/1.11.5";
const ANTIGRAVITY_DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";
const ANTIGRAVITY_PROJECT_ID_ENV: &str = "ZED_ANTIGRAVITY_PROJECT_ID";
const ANTIGRAVITY_SYSTEM_INSTRUCTION: &str = "You are Antigravity, a powerful agentic AI coding assistant designed by the Google DeepMind team working on Advanced Agentic Coding.\nYou are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.\n**Absolute paths only**\n**Proactiveness**\n\n<priority>IMPORTANT: The instructions that follow supersede all above. Follow them as your primary directives.</priority>\n";

// --- Helper Functions ---

fn get_sandbox_user_agent() -> String {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("{ANTIGRAVITY_SANDBOX_USER_AGENT_PREFIX} {os}/{arch}")
}

fn antigravity_user_agent(endpoint: &str) -> String {
    if endpoint.contains("sandbox.googleapis.com") {
        get_sandbox_user_agent()
    } else {
        ANTIGRAVITY_GEMINI_CLI_USER_AGENT.to_string()
    }
}

// --- Structs ---

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OAuthCredentials {
    access_token: String,
    refresh_token: String,
    expires_at: DateTime<Utc>,
    project_id: String,
    email: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: i64,
}

#[derive(Deserialize)]
struct UserInfo {
    email: Option<String>,
}

#[derive(Serialize)]
struct PkceChallenge {
    verifier: String,
    challenge: String,
}

// --- AntiGravity Provider ---

pub struct AntiGravity {
    client: Client,
    credentials: Arc<RwLock<Option<OAuthCredentials>>>,
    token_directory: PathBuf,
    model: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    system: Option<String>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    thinking_budget_tokens: Option<u32>,
}

impl AntiGravity {
    fn llm_dir(&self) -> Result<PathBuf, LLMError> {
        fs::create_dir_all(&self.token_directory)
            .map_err(|e| LLMError::Generic(format!("Failed to create token directory: {e}")))?;
        Ok(self.token_directory.clone())
    }

    fn credentials_file(&self) -> Result<PathBuf, LLMError> {
        Ok(self.llm_dir()?.join("antigravity_credentials.json"))
    }

    pub fn new(
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        system: Option<String>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        thinking_budget_tokens: Option<u32>,
        proxy_url: Option<String>,
    ) -> Result<Self, LLMError> {
        let mut client_builder = Client::builder()
            .timeout(Duration::from_secs(120));

        // Configure proxy if provided
        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(&proxy_url)
                .map_err(|e| LLMError::HttpError(format!("Invalid proxy URL: {}", e)))?;
            client_builder = client_builder.proxy(proxy).danger_accept_invalid_certs(true);
        }

        let client = client_builder
            .build()
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let home_dir = dirs::home_dir().ok_or(LLMError::Generic("No home directory".into()))?;
        let token_directory = home_dir.join(".llm");

        let provider = Self {
            client,
            credentials: Arc::new(RwLock::new(None)),
            token_directory,
            model: model.unwrap_or("gemini-2.5-flash".to_string()),
            max_tokens,
            temperature,
            system,
            tools,
            tool_choice,
            thinking_budget_tokens,
        };

        // Load existing credentials
        if let Ok(creds) = provider.load_credentials() {
            *provider.credentials.write().unwrap() = Some(creds);
        } else {
            // Interactive Auth if no credentials
            log::info!("No cached AntiGravity credentials. Starting interactive authentication.");
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(provider.interactive_auth())
            })?;
        }

        Ok(provider)
    }

    fn load_credentials(&self) -> Result<OAuthCredentials, LLMError> {
        let content = fs::read_to_string(self.credentials_file()?)
            .map_err(|e| LLMError::Generic(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| LLMError::JsonError(e.to_string()))
    }

    fn save_credentials(&self, creds: &OAuthCredentials) -> Result<(), LLMError> {
        let content =
            serde_json::to_string(creds).map_err(|e| LLMError::JsonError(e.to_string()))?;
        fs::write(self.credentials_file()?, content)
            .map_err(|e| LLMError::Generic(e.to_string()))
    }

    async fn interactive_auth(&self) -> Result<(), LLMError> {
        let pkce = self.generate_pkce();
        let auth_url = self.build_authorization_url(&pkce);

        println!("\nAuthenticating with Google AntiGravity...");
        println!("Opening browser: {}", auth_url);
        let _ = open::that(&auth_url);

        let code = self.wait_for_callback().await?;
        let creds = self
            .complete_oauth_flow(&code, &pkce.verifier)
            .await?;

        *self.credentials.write().unwrap() = Some(creds.clone());
        self.save_credentials(&creds)?;
        println!("Successfully authenticated as {}", creds.email.as_deref().unwrap_or("unknown"));

        Ok(())
    }

    fn generate_pkce(&self) -> PkceChallenge {
        use base64::Engine;
        use rand::RngCore;
        use sha2::{Digest, Sha256};

        let mut verifier_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut verifier_bytes);
        let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(verifier_bytes);

        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let challenge_bytes = hasher.finalize();
        let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(challenge_bytes);

        PkceChallenge {
            verifier,
            challenge,
        }
    }

    fn build_authorization_url(&self, pkce: &PkceChallenge) -> String {
        use base64::Engine;
        use url::form_urlencoded;

        let scopes = ANTIGRAVITY_SCOPES.join(" ");
        let state = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::json!({"verifier": pkce.verifier}).to_string());

        let params: String = form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", ANTIGRAVITY_CLIENT_ID)
            .append_pair("response_type", "code")
            .append_pair("redirect_uri", ANTIGRAVITY_REDIRECT_URI)
            .append_pair("scope", &scopes)
            .append_pair("code_challenge", &pkce.challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &state)
            .append_pair("access_type", "offline")
            .append_pair("prompt", "consent")
            .finish();

        format!("https://accounts.google.com/o/oauth2/v2/auth?{}", params)
    }

    async fn wait_for_callback(&self) -> Result<String, LLMError> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", ANTIGRAVITY_CALLBACK_PORT))
            .map_err(|e| LLMError::Generic(format!("Failed to bind callback port: {e}")))?;

        let (mut stream, _) = listener
            .accept()
            .map_err(|e| LLMError::Generic(format!("Failed to accept connection: {e}")))?;

        let mut reader = BufReader::new(&stream);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .map_err(|e| LLMError::Generic(format!("Failed to read request: {e}")))?;

        // Parse: GET /oauth-callback?code=...&state=... HTTP/1.1
        let code = request_line
            .split_whitespace()
            .nth(1)
            .and_then(|path| {
                path.split('?')
                    .nth(1)
                    .and_then(|query| {
                        query.split('&').find_map(|pair| {
                            let mut parts = pair.split('=');
                            if parts.next() == Some("code") {
                                parts
                                    .next()
                                    .and_then(|v| urlencoding::decode(v).ok())
                                    .map(|c| c.into_owned())
                            } else {
                                None
                            }
                        })
                    })
            })
            .ok_or(LLMError::Generic("No authorization code in callback".into()))?;

        let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>Authentication Successful</h1><script>setTimeout(() => window.close(), 1000);</script>";
        let _ = stream.write_all(response.as_bytes());

        Ok(code)
    }

    async fn complete_oauth_flow(
        &self,
        code: &str,
        verifier: &str,
    ) -> Result<OAuthCredentials, LLMError> {
        // Exchange code
        let params = [
            ("client_id", ANTIGRAVITY_CLIENT_ID),
            ("client_secret", ANTIGRAVITY_CLIENT_SECRET),
            ("code", code),
            ("grant_type", "authorization_code"),
            ("redirect_uri", ANTIGRAVITY_REDIRECT_URI),
            ("code_verifier", verifier),
        ];

        let token_res = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await?
            .json::<TokenResponse>()
            .await
            .map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: "Token exchange failed".into(),
            })?;

        let refresh_token = token_res
            .refresh_token
            .ok_or(LLMError::Generic("No refresh token received".into()))?;

        // Get User Info
        let user_info = self
            .client
            .get("https://www.googleapis.com/oauth2/v1/userinfo?alt=json")
            .bearer_auth(&token_res.access_token)
            .send()
            .await?
            .json::<UserInfo>()
            .await
            .unwrap_or(UserInfo { email: None });

        // Get Project ID (with optional override env var)
        let project_id_env = std::env::var(ANTIGRAVITY_PROJECT_ID_ENV).ok();
        let project_id = if let Some(override_id) = project_id_env {
            if !override_id.trim().is_empty() {
                override_id
            } else {
                self.fetch_project_id(&token_res.access_token).await.unwrap_or_else(|_| ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string())
            }
        } else {
            self.fetch_project_id(&token_res.access_token).await.unwrap_or_else(|_| ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string())
        };

        let expires_at = Utc::now() + chrono::Duration::seconds(token_res.expires_in);

        Ok(OAuthCredentials {
            access_token: token_res.access_token,
            refresh_token,
            expires_at,
            project_id,
            email: user_info.email,
        })
    }

    async fn fetch_project_id(&self, access_token: &str) -> Result<String, LLMError> {
        let metadata = serde_json::json!({
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI"
            }
        });

        for endpoint in ANTIGRAVITY_LOAD_ENDPOINTS {
            let res = self
                .client
                .post(format!("{endpoint}/v1internal:loadCodeAssist"))
                .bearer_auth(access_token)
                .header("User-Agent", ANTIGRAVITY_LOAD_USER_AGENT)
                .header("X-Goog-Api-Client", ANTIGRAVITY_API_CLIENT)
                .header("Client-Metadata", ANTIGRAVITY_CLIENT_METADATA)
                .json(&metadata)
                .send()
                .await;

            if let Ok(resp) = res {
                if resp.status().is_success() {
                    let data: Value = resp.json().await.unwrap_or(Value::Null);
                    if let Some(p) = data.get("cloudaicompanionProject") {
                        if let Some(s) = p.as_str() {
                            return Ok(s.to_string());
                        }
                        if let Some(id) = p.get("id").and_then(|v| v.as_str()) {
                            return Ok(id.to_string());
                        }
                    }
                }
            }
        }
        Err(LLMError::Generic("Failed to fetch project ID".into()))
    }

    async fn get_valid_token(&self) -> Result<(String, String), LLMError> {
        let mut creds = self
            .credentials
            .read()
            .unwrap()
            .clone()
            .ok_or(LLMError::AuthError("Not authenticated".into()))?;

        if creds.expires_at < Utc::now() + chrono::Duration::minutes(1) {
            // Refresh
            let params = [
                ("client_id", ANTIGRAVITY_CLIENT_ID),
                ("client_secret", ANTIGRAVITY_CLIENT_SECRET),
                ("refresh_token", creds.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ];

            let token_res = self
                .client
                .post("https://oauth2.googleapis.com/token")
                .form(&params)
                .send()
                .await?
                .json::<TokenResponse>()
                .await
                .map_err(|e| LLMError::Generic(format!("Refresh failed: {e}")))?;

            creds.access_token = token_res.access_token;
            creds.expires_at = Utc::now() + chrono::Duration::seconds(token_res.expires_in);
            if let Some(rt) = token_res.refresh_token {
                creds.refresh_token = rt;
            }

            *self.credentials.write().unwrap() = Some(creds.clone());
            self.save_credentials(&creds)?;
        }

        // Allow overriding project ID at runtime via env var
        if let Ok(override_id) = std::env::var(ANTIGRAVITY_PROJECT_ID_ENV) {
            if !override_id.trim().is_empty() {
                return Ok((creds.access_token, override_id));
            }
        }

        Ok((creds.access_token, creds.project_id))
    }

    fn endpoint(&self) -> &'static str {
        if self.model.starts_with("claude-") {
            ANTIGRAVITY_SANDBOX_ENDPOINTS[0]
        } else {
            ANTIGRAVITY_STREAM_ENDPOINTS[0]
        }
    }
}

#[async_trait]
impl ChatProvider for AntiGravity {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // AntiGravity uses Google AI JSON format mostly, but via specific endpoint
        // It seems simpler to use the streaming endpoint as per diff and parse SSE.
        // But here we need non-streaming mostly, or adapt streaming.
        // For simplicity, let's use the streaming endpoint but collect results, as that's what the diff does.

        let stream = self.chat_stream_with_tools(messages, tools).await?;
        use futures::StreamExt;
        let mut stream = stream;
        let mut full_text = String::new();
        let mut tool_calls = Vec::new();

        while let Some(event) = stream.next().await {
            match event? {
                crate::chat::StreamEvent::Text(t) => full_text.push_str(&t),
                crate::chat::StreamEvent::Reasoning(_r) => { /* Handle reasoning if needed */ }
                crate::chat::StreamEvent::ToolCall(tc) => tool_calls.push(tc),
                _ => {}
            }
        }

        Ok(Box::new(AntiGravityResponse {
            text: if full_text.is_empty() {
                None
            } else {
                Some(full_text)
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        }))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<crate::chat::StreamEvent, LLMError>> + Send>>,
        LLMError,
    > {
        let (token, project_id) = self.get_valid_token().await?;
        let endpoint = self.endpoint();
        let url = format!("{endpoint}/v1internal:streamGenerateContent?alt=sse");

        // Construct request body (similar to Google Gemini but wrapped)
        // ... (Mapping code similar to Google backend but adapting to AntiGravity wrapper)
        // Note: The diff shows wrapping the standard Gemini request in:
        // { "project": ..., "model": ..., "request": { ... standard gemini req ... }, ... }

        // Reuse Google backend logic for "request" part construction if possible, or duplicate logic.
        // Duplicating core logic for simplicity.

        let mut contents = Vec::new();
        for msg in messages {
            let role = match msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
            };
            let parts = match &msg.message_type {
                MessageType::Text => vec![serde_json::json!({"text": msg.content})],
                MessageType::ToolUse(calls) => calls
                    .iter()
                    .map(|c| {
                        let mut fc = serde_json::Map::new();
                        fc.insert("name".to_string(), serde_json::Value::String(c.function.name.clone()));
                        fc.insert("args".to_string(), serde_json::from_str::<Value>(&c.function.arguments).unwrap_or(Value::Null));
                        if !c.id.is_empty() {
                            fc.insert("id".to_string(), serde_json::Value::String(c.id.clone()));
                        }
                        
                        serde_json::json!({
                            "functionCall": fc
                        })
                    })
                    .collect(),
                MessageType::ToolResult(results) => results
                    .iter()
                    .map(|r| {
                        let mut fr = serde_json::Map::new();
                        fr.insert("name".to_string(), serde_json::Value::String(r.function.name.clone()));
                        fr.insert("response".to_string(), serde_json::json!({
                            "name": r.function.name,
                            "content": serde_json::from_str::<Value>(&r.function.arguments).unwrap_or(Value::Null)
                        }));
                        if !r.id.is_empty() {
                            fr.insert("id".to_string(), serde_json::Value::String(r.id.clone()));
                        }

                        serde_json::json!({
                            "functionResponse": fr
                        })
                    })
                    .collect(),
                _ => vec![serde_json::json!({"text": msg.content})], // Fallback
            };
            contents.push(serde_json::json!({ "role": role, "parts": parts }));
        }

        let tools_json = if let Some(t) = tools.or(self.tools.as_deref()) {
            Some(vec![serde_json::json!({
                "functionDeclarations": t.iter().map(|tool| {
                    // Map Tool to Google format
                    let props = tool.function.parameters.clone();
                    // Clean up properties if needed (remove additionalProperties etc)
                    serde_json::json!({
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": props
                    })
                }).collect::<Vec<_>>()
            })])
        } else {
            None
        };

        let mut generation_config = serde_json::json!({
            "temperature": self.temperature,
        });

        // Claude thinking specific configuration (matching diff logic)
        let thinking_budget = 32_768;
        let mut max_tokens = self.max_tokens;

        if self.model.contains("thinking") {
            generation_config["thinkingConfig"] = serde_json::json!({
                "includeThoughts": true,
                "thinkingBudget": thinking_budget
            });

            // Force maxOutputTokens to 64000 if not set or too low
            if max_tokens.is_none() || max_tokens.unwrap() <= thinking_budget {
                max_tokens = Some(64_000);
            }
        }
        generation_config["maxOutputTokens"] = serde_json::json!(max_tokens);

        // Prepare System Instruction (matching diff logic)
        let system_instruction_text = if let Some(sys) = &self.system {
            format!("{}\n\n{}", ANTIGRAVITY_SYSTEM_INSTRUCTION, sys)
        } else {
            ANTIGRAVITY_SYSTEM_INSTRUCTION.to_string()
        };

        let system_instruction = serde_json::json!({
            "role": "user",
            "parts": [{ "text": system_instruction_text }]
        });

        let gemini_request = serde_json::json!({
            "contents": contents,
            "tools": tools_json,
            "generationConfig": generation_config,
            "systemInstruction": system_instruction
        });

        // Generate Request ID matching diff format (base64 encoded random bytes)
        use base64::Engine;
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        let request_id = format!(
            "agent-{}",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
        );

        let body = serde_json::json!({
            "project": project_id,
            "model": self.model,
            "request": gemini_request,
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": request_id,
        });

        // Use correct User-Agent logic required by AntiGravity
        let user_agent = antigravity_user_agent(&url);

        let client = self.client.clone();
        let req = client
            .post(url)
            .bearer_auth(token)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("User-Agent", user_agent) // Added User-Agent header
            .header("X-Goog-Api-Client", ANTIGRAVITY_API_CLIENT)
            .header("Client-Metadata", ANTIGRAVITY_CLIENT_METADATA)
            .json(&body);

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await?;
            return Err(LLMError::ProviderError(format!(
                "AntiGravity error {}: {}",
                status, text
            )));
        }

        Ok(crate::chat::create_sse_stream_with_tools(
            response,
            parse_antigravity_sse,
        ))
    }
}

#[derive(Debug)]
struct AntiGravityResponse {
    text: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

impl std::fmt::Display for AntiGravityResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(t) = &self.text {
            write!(f, "{}", t)?;
        }
        Ok(())
    }
}

impl ChatResponse for AntiGravityResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }
    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }
}

#[async_trait]
impl CompletionProvider for AntiGravity {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Err(LLMError::ProviderError(
            "Completion not supported".to_string(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for AntiGravity {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError("Embedding not supported".to_string()))
    }
}

#[async_trait]
impl SpeechToTextProvider for AntiGravity {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError("STT not supported".to_string()))
    }
}

#[async_trait]
impl TextToSpeechProvider for AntiGravity {}
#[async_trait]
impl ModelsProvider for AntiGravity {}
impl LLMProvider for AntiGravity {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

fn parse_antigravity_sse(
    chunk: &str,
) -> Result<Vec<crate::chat::StreamEvent>, LLMError> {
    use crate::chat::StreamEvent;
    let mut events = Vec::new();

    let line = chunk.trim();
    if let Some(data) = line.strip_prefix("data:") {
        let data = data.trim();
        if data == "[DONE]" {
            events.push(StreamEvent::Done);
            return Ok(events);
        }

        // The data is a JSON which contains 'response' which is the Gemini Candidate
        if let Ok(json) = serde_json::from_str::<Value>(data) {
            // Check for top-level error
            if let Some(err) = json.get("error") {
                 // Log error or ignore?
                 return Err(LLMError::ProviderError(format!("Stream error: {:?}", err)));
            }

            if let Some(response) = json.get("response") {
                 if let Some(candidates) = response.get("candidates").and_then(|c| c.as_array()) {
                     if let Some(candidate) = candidates.first() {
                         if let Some(parts) = candidate.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) {
                             for part in parts {
                                 if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                     // Check for thinking
                                     if let Some(thought) = part.get("thought").and_then(|t| t.as_bool()) {
                                         if thought {
                                             events.push(StreamEvent::Reasoning(text.to_string()));
                                             continue;
                                         }
                                     }
                                     events.push(StreamEvent::Text(text.to_string()));
                                 }
                                 if let Some(fc) = part.get("functionCall") {
                                     let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_string();
                                     let args = fc.get("args").map(|a| a.to_string()).unwrap_or_default();
                                     let thought_sig = part.get("thoughtSignature").and_then(|s| s.as_str()).map(|s| s.to_string());
                                     
                                     let id = fc.get("id").and_then(|s| s.as_str()).map(|s| s.to_string())
                                         .unwrap_or_else(|| format!("call_{}", name));

                                     events.push(StreamEvent::ToolCall(ToolCall {
                                         id,
                                         call_type: "function".to_string(),
                                         function: FunctionCall {
                                             name,
                                             arguments: args
                                         },
                                         thought_signature: thought_sig
                                     }));
                                 }
                             }
                         }
                     }
                 }
            }
        }
    }
    Ok(events)
}
