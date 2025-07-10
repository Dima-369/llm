use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, Tool, ToolChoice, Usage},
    chat::{ChatRole, MessageType},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    secret_store::SecretStore,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider, ToolCall,
};
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
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
}

impl LLMProvider for Together {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

impl Together {
    pub fn new(
        proxy_url: Option<String>,
        _secrets: &SecretStore,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        let mut builder = reqwest::Client::builder();

        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(&proxy_url).expect("Failed to create proxy");
            builder = builder.proxy(proxy).danger_accept_invalid_certs(true);
        }

        let client = builder.build().expect("Failed to build client");

        Self {
            api_key: Arc::new(Mutex::new(String::new())),
            client,
            tools,
            tool_choice,
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
                "Failed to get activation key: Status {status} - {body}"
            )));
        }

        let json: Value = serde_json::from_str(&body).map_err(|e| {
            LLMError::Generic(format!("Failed to parse activation key response: {e}"))
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
            .header("Authorization", format!("Bearer {api_key}"))
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
                "API call failed with status: {status}, body: {res_body}"
            )));
        }

        let response_body: ResponseBody = res
            .json()
            .await
            .map_err(|e| LLMError::Generic(format!("Failed to parse response body: {e}")))?;

        Ok(CompletionResponse::from(response_body))
    }
}

#[async_trait]
impl ChatProvider for Together {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[crate::chat::Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let api_key = self.get_activation_key().await?;

        let mut together_msgs: Vec<TogetherChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    together_msgs.push(
                        // Clone strings to own them
                        TogetherChatMessage {
                            role: "tool",
                            tool_call_id: Some(Box::leak(result.id.clone().into_boxed_str())),
                            tool_calls: None,
                            content: Some(Box::leak(
                                result.function.arguments.clone().into_boxed_str(),
                            )),
                        },
                    );
                }
            } else {
                together_msgs.push(chat_message_to_api_message(msg.clone()))
            }
        }

        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());

        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let body = RequestBody {
            messages: together_msgs,
            model: DEFAULT_MODEL.to_string(),
            stream: false,
            temperature: None,
            max_tokens: None,
            tools: request_tools,
            tool_choice: request_tool_choice,
        };

        let res = self
            .client
            .post(CHAT_COMPLETIONS_ENDPOINT)
            .header("Authorization", format!("Bearer {api_key}"))
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
                "API call failed with status: {status}, body: {res_body}"
            )));
        }

        let response_body: ResponseBody = res
            .json()
            .await
            .map_err(|e| LLMError::Generic(format!("Failed to parse response body: {e}")))?;

        Ok(Box::new(TogetherChatResponse {
            text: response_body
                .choices
                .first()
                .and_then(|c| c.message.content.clone()),
            tool_calls: response_body
                .choices
                .first()
                .and_then(|c| c.message.tool_calls.clone()),
            usage: response_body.usage,
        }))
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
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
struct TogetherChatMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TogetherFunctionCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

#[derive(Serialize, Debug)]
struct TogetherFunctionPayload<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize, Debug)]
struct TogetherFunctionCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    content_type: &'a str,
    function: TogetherFunctionPayload<'a>,
}

#[derive(Serialize, Debug)]
struct RequestBody {
    messages: Vec<TogetherChatMessage<'static>>,
    model: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

impl From<&CompletionRequest> for RequestBody {
    fn from(request: &CompletionRequest) -> Self {
        Self {
            messages: vec![TogetherChatMessage {
                role: "user",
                content: Some(Box::leak(request.prompt.clone().into_boxed_str())),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: DEFAULT_MODEL.to_string(),
            stream: false,
            temperature: request.temperature,

            max_tokens: request.max_tokens,
            tools: None,
            tool_choice: None,
        }
    }
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct ResponseBody {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Usage,
}

impl From<ResponseBody> for CompletionResponse {
    fn from(response: ResponseBody) -> Self {
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        Self { text }
    }
}

#[derive(Debug)]
pub struct TogetherChatResponse {
    text: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    usage: Usage,
}

impl ChatResponse for TogetherChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }

    fn usage(&self) -> Option<Usage> {
        Some(self.usage.clone())
    }
}

impl fmt::Display for TogetherChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.text, &self.tool_calls) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

// Create an owned TogetherChatMessage that doesn't borrow from any temporary variables
fn chat_message_to_api_message(chat_msg: ChatMessage) -> TogetherChatMessage<'static> {
    // For other message types, create an owned TogetherChatMessage
    TogetherChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        },
        tool_call_id: None,
        content: Some(Box::leak(chat_msg.content.into_boxed_str())),
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                let owned_calls: Vec<TogetherFunctionCall<'static>> = calls
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

                        TogetherFunctionCall {
                            id: id_str,
                            content_type: "function",
                            function: TogetherFunctionPayload {
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
