//! Cohere API client implementation for chat and completion functionality.
//! 
//! This module provides integration with Cohere's models through their API.



use crate::{
    builder::LLMBackend,
    chat::Tool,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use crate::{
    chat::{ChatResponse, ToolChoice},
    ToolCall,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use either::*;
use futures::stream::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for interacting with Cohere's API.
/// 
/// Provides methods for chat and completion requests using Cohere's models.
pub struct Cohere {
    pub api_key: String,
    pub base_url: Url,
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
    /// Embedding parameters
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub reasoning_effort: Option<String>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    client: Client,
}

/// Individual message in an Cohere chat conversation.
#[derive(Serialize, Debug)]
struct CohereChatMessage<'a> {
    #[allow(dead_code)]
    role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<MessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereFunctionCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
struct CohereFunctionPayload<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize, Debug)]
struct CohereFunctionCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    content_type: &'a str,
    function: CohereFunctionPayload<'a>,
}

#[derive(Serialize, Debug)]
struct MessageContent<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<&'a str>,
}

/// Individual image message in an Cohere chat conversation.
#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

#[derive(Serialize)]
struct CohereEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// Request payload for Cohere's chat API endpoint.
#[derive(Serialize, Debug)]
struct CohereChatRequest<'a> {
    model: &'a str,
    messages: Vec<CohereChatMessage<'a>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<CohereResponseFormat>,
}



/// Response from Cohere's chat API endpoint.
#[derive(Deserialize, Debug)]
struct CohereChatResponse {
    choices: Vec<CohereChatChoice>,
}

/// Individual choice within an Cohere chat API response.
#[derive(Deserialize, Debug)]
struct CohereChatChoice {
    message: CohereChatMsg,
}

/// Message content within an Cohere chat API response.
#[derive(Deserialize, Debug)]
struct CohereChatMsg {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
struct CohereEmbeddingData {
    embedding: Vec<f32>,
}
#[derive(Deserialize, Debug)]
struct CohereEmbeddingResponse {
    data: Vec<CohereEmbeddingData>,
}

/// Response from Cohere's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct CohereChatStreamResponse {
    choices: Vec<CohereChatStreamChoice>,
}

/// Individual choice within an Cohere streaming chat API response.
#[derive(Deserialize, Debug)]
struct CohereChatStreamChoice {
    delta: CohereChatStreamDelta,
}

/// Delta content within an Cohere streaming chat API response.
#[derive(Deserialize, Debug)]
struct CohereChatStreamDelta {
    content: Option<String>,
}

/// An object specifying the format that the model must output.
///Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](https://platform.cohere.com/docs/guides/structured-outputs).
/// Setting to `{ "type": "json_object" }` enables the older JSON mode, which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for models that support it.
#[derive(Deserialize, Debug, Serialize)]
enum CohereResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
struct CohereResponseFormat {
    #[serde(rename = "type")]
    response_type: CohereResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

impl From<StructuredOutputFormat> for CohereResponseFormat {
    /// Modify the schema to ensure that it meets Cohere's requirements.
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        // It's possible to pass a StructuredOutputJsonSchema without an actual schema.
        // In this case, just pass the StructuredOutputJsonSchema object without modifying it.
        match structured_response_format.schema {
            None => CohereResponseFormat {
                response_type: CohereResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                // Although [Cohere's specifications](https://platform.cohere.com/docs/guides/structured-outputs?api-mode=chat#additionalproperties-false-must-always-be-set-in-objects) say that the "additionalProperties" field is required, my testing shows that it is not.
                // Just to be safe, add it to the schema if it is missing.
                schema = if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                    schema
                } else {
                    schema
                };

                CohereResponseFormat {
                    response_type: CohereResponseType::JsonSchema,
                    json_schema: Some(StructuredOutputFormat {
                        name: structured_response_format.name,
                        description: structured_response_format.description,
                        schema: Some(schema),
                        strict: structured_response_format.strict,
                    }),
                }
            }
        }
    }
}

impl ChatResponse for CohereChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }
}

impl std::fmt::Display for CohereChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (
            &self.choices.first().unwrap().message.content,
            &self.choices.first().unwrap().message.tool_calls,
        ) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{}", tool_call)?;
                }
                write!(f, "{}", content)
            }
            (Some(content), None) => write!(f, "{}", content),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{}", tool_call)?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl Cohere {
    /// Creates a new Cohere client with the specified configuration.
    /// 
    /// # Arguments
    /// 
    /// * `api_key` - Cohere API key
    /// * `model` - Model to use (defaults to "gpt-3.5-turbo")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    /// * `embedding_encoding_format` - Format for embedding outputs
    /// * `embedding_dimensions` - Dimensions for embedding vectors
    /// * `tools` - Function tools that the model can use
    /// * `tool_choice` - Determines how the model uses tools
    /// * `reasoning_effort` - Reasoning effort level
    /// * `json_schema` - JSON schema for structured output
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        proxy_url: Option<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
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
            api_key: api_key.into(),
            base_url: Url::parse(
                &base_url.unwrap_or_else(|| "https://api.cohere.ai/compatibility/v1/".to_owned()),
            )
            .expect("Failed to prase base Url"),
            model: model.unwrap_or("command-r-plus".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            reasoning_effort,
            json_schema,
        }
    }
}

#[async_trait]
impl ChatProvider for Cohere {
    /// Sends a chat request to Cohere's API.
    /// 
    /// # Arguments
    /// 
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    /// # Returns
    /// 
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".to_string()));
        }

        // Clone the messages to have an owned mutable vector.
        let messages = messages.to_vec();

        let mut cohere_msgs: Vec<CohereChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    cohere_msgs.push(
                        // Clone strings to own them
                        CohereChatMessage {
                            role: "tool",
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        },
                    );
                }
            } else {
                cohere_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            cohere_msgs.insert(
                0,
                CohereChatMessage {
                    role: "developer",
                    content: Some(Left(vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(system),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let response_format:
            Option<CohereResponseFormat> = self.json_schema.clone().map(|s| s.into());

        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());

        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let body = CohereChatRequest {
            model: &self.model,
            messages: cohere_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
            response_format,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Cohere request payload: {}", json);
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        log::debug!("Cohere HTTP status: {}", response.status());

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let raw_response = response.text().await?;
            return Err(LLMError::TooManyRequests(raw_response));
        }

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Cohere API returned error status: {}", status),
                raw_response: error_text,
            });
        }

        // Parse the successful response
        let resp_text = response.text().await?;
        let json_resp: Result<CohereChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode Cohere API response: {}", e),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to Cohere's API.
    /// 
    /// # Arguments
    /// 
    /// * `messages` - Slice of chat messages representing the conversation
    /// 
    /// # Returns
    /// 
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".to_string()));
        }

        let messages = messages.to_vec();
        let mut cohere_msgs: Vec<CohereChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    cohere_msgs.push(CohereChatMessage {
                        role: "tool",
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                cohere_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            cohere_msgs.insert(
                0,
                CohereChatMessage {
                    role: "developer",
                    content: Some(Left(vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(system),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let body = CohereChatRequest {
            model: &self.model,
            messages: cohere_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            response_format: None,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Cohere API returned error status: {}", status),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(response, parse_sse_chunk))
    }
}

// Create an owned CohereChatMessage that doesn't borrow from any temporary variables
fn chat_message_to_api_message(chat_msg: ChatMessage) -> CohereChatMessage<'static> {
    // For other message types, create an owned CohereChatMessage
    CohereChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        },
        tool_call_id: None,
        content: match &chat_msg.message_type {
            MessageType::Text => Some(Right(chat_msg.content.clone())),
            // Image case is handled separately above
            MessageType::Image(_) => unreachable!(),
            MessageType::Pdf(_) => unimplemented!(),
            MessageType::ImageURL(url) => {
                // Clone the URL to create an owned version
                let owned_url = url.clone();
                // Leak the string to get a 'static reference
                let url_str = Box::leak(owned_url.into_boxed_str());
                Some(Left(vec![MessageContent {
                    message_type: Some("image_url"),
                    text: None,
                    image_url: Some(ImageUrlContent { url: url_str }),
                    tool_output: None,
                    tool_call_id: None,
                }]))
            }
            MessageType::ToolUse(_) => None,
            MessageType::ToolResult(_) => None,
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                let owned_calls: Vec<CohereFunctionCall<'static>> = calls
                    .iter()
                    .map(|c| {
                        let owned_id = c.id.clone();
                        let owned_name = c.function.name.clone();
                        let owned_args = c.function.arguments.clone();

                        // Need to leak these strings to create 'static references
                        // This is a deliberate choice to solve the lifetime issue
                        // The small memory leak is acceptable in this context
                        let id_str = Box::leak(owned_id.into_boxed_str());
                        let name_str = Box::leak(owned_name.into_boxed_str());
                        let args_str = Box::leak(owned_args.into_boxed_str());

                        CohereFunctionCall {
                            id: id_str,
                            content_type: "function",
                            function: CohereFunctionPayload {
                                name: name_str,
                                arguments: args_str,
                            },
                        }
                    })
                    .collect();
                Some(owned_calls)
            }
            _ => None,
        },
    }
}

#[async_trait]
impl CompletionProvider for Cohere {
    /// Sends a completion request to Cohere's API.
    /// 
    /// Currently not implemented.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Cohere completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for Cohere {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for Cohere {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".into()));
        }

        let emb_format = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = CohereEmbeddingRequest {
            model: self.model.clone(),
            input,
            encoding_format: Some(emb_format),
            dimensions: self.embedding_dimensions,
        };

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: CohereEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CohereModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for CohereModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|t| chrono::DateTime::from_timestamp(t as i64, 0).unwrap_or_default())
            .unwrap_or_default()
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CohereModelListResponse {
    pub data: Vec<CohereModelEntry>,
}

impl ModelListResponse for CohereModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|e| e.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::Cohere
    }
}

#[async_trait]
impl ModelsProvider for Cohere {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .get(url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = resp.json::<CohereModelListResponse>().await?;

        Ok(Box::new(result))
    }
}

impl LLMProvider for Cohere {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl TextToSpeechProvider for Cohere {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not implement text to speech endpoint yet.".into(),
        ))
    }
}

/// Parses a Server-Sent Events (SSE) chunk from Cohere's streaming API.
/// 
/// # Arguments
/// 
/// * `chunk` - The raw SSE chunk text
/// 
/// # Returns
/// 
/// * `Ok(Some(String))` - Content token if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
fn parse_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    let mut collected_content = String::new();

    for line in chunk.lines() {
        let line = line.trim();

        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                if collected_content.is_empty() {
                    return Ok(None);
                } else {
                    return Ok(Some(collected_content));
                }
            }

            match serde_json::from_str::<CohereChatStreamResponse>(data) {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
    }

    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(collected_content))
    }
}
