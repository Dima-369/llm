use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::LLMError, ToolCall};

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatRole {
    /// The user/human participant in the conversation
    User,
    /// The AI assistant participant in the conversation
    Assistant,
}

/// The supported MIME type of an image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ImageMime {
    /// JPEG image
    JPEG,
    /// PNG image
    PNG,
    /// GIF image
    GIF,
    /// WebP image
    WEBP,
}

impl ImageMime {
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageMime::JPEG => "image/jpeg",
            ImageMime::PNG => "image/png",
            ImageMime::GIF => "image/gif",
            ImageMime::WEBP => "image/webp",
        }
    }
}

/// The type of a message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MessageType {
    /// A text message
    #[default]
    Text,
    /// An image message
    Image((ImageMime, Vec<u8>)),
    /// PDF message
    Pdf(Vec<u8>),
    /// An image URL message
    ImageURL(String),
    /// A tool use
    ToolUse(Vec<ToolCall>),
    /// Tool result
    ToolResult(Vec<ToolCall>),
}

/// The type of reasoning effort for a message in a chat conversation.
pub enum ReasoningEffort {
    /// Low reasoning effort
    Low,
    /// Medium reasoning effort
    Medium,
    /// High reasoning effort
    High,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of who sent this message (user or assistant)
    pub role: ChatRole,
    /// The type of the message (text, image, audio, video, etc)
    pub message_type: MessageType,
    /// The text content of the message
    pub content: String,
}

/// Represents a parameter in a function tool
#[derive(Debug, Clone, Serialize)]
pub struct ParameterProperty {
    /// The type of the parameter (e.g. "string", "number", "array", etc)
    #[serde(rename = "type")]
    pub property_type: String,
    /// Description of what the parameter does
    pub description: String,
    /// When type is "array", this defines the type of the array items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParameterProperty>>,
    /// When type is "enum", this defines the possible values for the parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_list: Option<Vec<String>>,
}

/// Represents the parameters schema for a function tool
#[derive(Debug, Clone, Serialize)]
pub struct ParametersSchema {
    /// The type of the parameters object (usually "object")
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Map of parameter names to their properties
    pub properties: HashMap<String, ParameterProperty>,
    /// List of required parameter names
    pub required: Vec<String>,
}

/// Represents a function definition for a tool.
///
/// The `parameters` field stores the JSON Schema describing the function
/// arguments.  It is kept as a raw `serde_json::Value` to allow arbitrary
/// complexity (nested arrays/objects, `oneOf`, etc.) without requiring a
/// bespoke Rust structure.
///
/// Builder helpers can still generate simple schemas automatically, but the
/// user may also provide any valid schema directly.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionTool {
    /// Name of the function
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema describing the parameters
    pub parameters: Value,
}

/// Defines rules for structured output responses based on [OpenAI's structured output requirements](https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format).
/// Individual providers may have additional requirements or restrictions, but these should be handled by each provider's backend implementation.
///
/// If you plan on deserializing into this struct, make sure the source text has a `"name"` field, since that's technically the only thing required by OpenAI.
///
/// ## Example
///
/// ```
/// use llm::chat::StructuredOutputFormat;
/// use serde_json::json;
///
/// let response_format = r#"
///     {
///         "name": "Student",
///         "description": "A student object",
///         "schema": {
///             "type": "object",
///             "properties": {
///                 "name": {
///                     "type": "string"
///                 },
///                 "age": {
///                     "type": "integer"
///                 },
///                 "is_student": {
///                     "type": "boolean"
///                 }
///             },
///             "required": ["name", "age", "is_student"]
///         }
///     }
/// "#;
/// let structured_output: StructuredOutputFormat = serde_json::from_str(response_format).unwrap();
/// assert_eq!(structured_output.name, "Student");
/// assert_eq!(structured_output.description, Some("A student object".to_string()));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]

pub struct StructuredOutputFormat {
    /// Name of the schema
    pub name: String,
    /// The description of the schema
    pub description: Option<String>,
    /// The JSON schema for the structured output
    pub schema: Option<Value>,
    /// Whether to enable strict schema adherence
    pub strict: Option<bool>,
}

/// Represents a tool that can be used in chat
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    /// The type of tool (e.g. "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition if this is a function tool
    pub function: FunctionTool,
}

/// Tool choice determines how the LLM uses available tools.
/// The behavior is standardized across different LLM providers.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Model can use any tool, but it must use at least one.
    /// This is useful when you want to force the model to use tools.
    Any,

    /// Model can use any tool, and may elect to use none.
    /// This is the default behavior and gives the model flexibility.
    #[default]
    Auto,

    /// Model must use the specified tool and only the specified tool.
    /// The string parameter is the name of the required tool.
    /// This is useful when you want the model to call a specific function.
    Tool(String),

    /// Explicitly disables the use of tools.
    /// The model will not use any tools even if they are provided.
    None,
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ToolChoice::Any => serializer.serialize_str("required"),
            ToolChoice::Auto => serializer.serialize_str("auto"),
            ToolChoice::None => serializer.serialize_str("none"),
            ToolChoice::Tool(name) => {
                use serde::ser::SerializeMap;

                // For tool_choice: {"type": "function", "function": {"name": "function_name"}}
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "function")?;

                // Inner function object
                let mut function_obj = std::collections::HashMap::new();
                function_obj.insert("name", name.as_str());

                map.serialize_entry("function", &function_obj)?;
                map.end()
            }
        }
    }
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct Usage {
    pub total_tokens: u32,
}

pub trait ChatResponse: std::fmt::Debug + std::fmt::Display {
    fn text(&self) -> Option<String>;
    fn tool_calls(&self) -> Option<Vec<ToolCall>>;
    fn thinking(&self) -> Option<String> {
        None
    }
    fn usage(&self) -> Option<Usage> {
        None
    }
}

/// Trait for providers that support chat-style interactions.
#[async_trait]
pub trait ChatProvider: Sync + Send {
    /// Sends a chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a chat request to the provider with a sequence of messages and tools.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError>;

    /// Sends a streaming chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        Err(LLMError::Generic(
            "Streaming not supported for this provider".to_string(),
        ))
    }

    /// Get current memory contents if provider supports memory
    async fn memory_contents(&self) -> Option<Vec<ChatMessage>> {
        None
    }

    /// Summarizes a conversation history into a concise 2-3 sentence summary
    ///
    /// # Arguments
    /// * `msgs` - The conversation messages to summarize
    ///
    /// # Returns
    /// A string containing the summary or an error if summarization fails
    async fn summarize_history(&self, msgs: &[ChatMessage]) -> Result<String, LLMError> {
        let prompt = format!(
            "Summarize in 2-3 sentences:\n{}",
            msgs.iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let req = [ChatMessage::user().content(prompt).build()];
        self.chat(&req)
            .await?
            .text()
            .ok_or(LLMError::Generic("no text in summary response".into()))
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
        }
    }
}

impl ChatMessage {
    /// Create a new builder for a user message
    pub fn user() -> ChatMessageBuilder {
        ChatMessageBuilder::new(ChatRole::User)
    }

    /// Create a new builder for an assistant message
    pub fn assistant() -> ChatMessageBuilder {
        ChatMessageBuilder::new(ChatRole::Assistant)
    }
}

/// Builder for ChatMessage
#[derive(Debug)]
pub struct ChatMessageBuilder {
    role: ChatRole,
    message_type: MessageType,
    content: String,
}

impl ChatMessageBuilder {
    /// Create a new ChatMessageBuilder with specified role
    pub fn new(role: ChatRole) -> Self {
        Self {
            role,
            message_type: MessageType::default(),
            content: String::new(),
        }
    }

    /// Set the message content
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = content.into();
        self
    }

    /// Set the message type as Image
    pub fn image(mut self, image_mime: ImageMime, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Image((image_mime, raw_bytes));
        self
    }

    /// Set the message type as Image
    pub fn pdf(mut self, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Pdf(raw_bytes);
        self
    }

    /// Set the message type as ImageURL
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.message_type = MessageType::ImageURL(url.into());
        self
    }

    /// Set the message type as ToolUse
    pub fn tool_use(mut self, tools: Vec<ToolCall>) -> Self {
        self.message_type = MessageType::ToolUse(tools);
        self
    }

    /// Set the message type as ToolResult
    pub fn tool_result(mut self, tools: Vec<ToolCall>) -> Self {
        self.message_type = MessageType::ToolResult(tools);
        self
    }

    /// Build the ChatMessage
    pub fn build(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            message_type: self.message_type,
            content: self.content,
        }
    }
}

/// Creates a Server-Sent Events (SSE) stream from an HTTP response.
///
/// # Arguments
///
/// * `response` - The HTTP response from the streaming API
/// * `parser` - Function to parse each SSE chunk into optional text content
///
/// # Returns
///
/// A pinned stream of text tokens or an error
pub(crate) fn create_sse_stream<F>(
    response: reqwest::Response,
    parser: F,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>
where
    F: Fn(&str) -> Result<Option<String>, LLMError> + Send + 'static,
{
    let stream = response
        .bytes_stream()
        .map(move |chunk| match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                parser(&text)
            }
            Err(e) => Err(LLMError::HttpError(e.to_string())),
        })
        .filter_map(|result| async move {
            match result {
                Ok(Some(content)) => Some(Ok(content)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });

    Box::pin(stream)
}

// Helper module for base64 encoding/decoding for serde
mod base64_helpers {
    use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = BASE64_STANDARD.encode(bytes);
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        BASE64_STANDARD.decode(s).map_err(serde::de::Error::custom)
    }
}

/// A version of MessageType designed for clean serialization.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SerializableMessageExtra {
    Image {
        mime: ImageMime,
        #[serde(with = "base64_helpers")]
        bytes: Vec<u8>,
    },
    Pdf {
        #[serde(with = "base64_helpers")]
        bytes: Vec<u8>,
    },
    ImageUrl {
        url: String,
    },
    ToolUse {
        calls: Vec<ToolCall>,
    },
    ToolResult {
        results: Vec<ToolCall>,
    },
}

/// A version of ChatMessage designed for clean serialization.
#[derive(Serialize, Deserialize, Debug)]
struct SerializableChatMessage {
    role: ChatRole,
    content: String,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    extra: Option<SerializableMessageExtra>,
}

impl From<ChatMessage> for SerializableChatMessage {
    fn from(msg: ChatMessage) -> Self {
        let extra = match msg.message_type {
            MessageType::Text => None,
            MessageType::Image((mime, bytes)) => {
                Some(SerializableMessageExtra::Image { mime, bytes })
            }
            MessageType::Pdf(bytes) => Some(SerializableMessageExtra::Pdf { bytes }),
            MessageType::ImageURL(url) => Some(SerializableMessageExtra::ImageUrl { url }),
            MessageType::ToolUse(calls) => Some(SerializableMessageExtra::ToolUse { calls }),
            MessageType::ToolResult(results) => {
                Some(SerializableMessageExtra::ToolResult { results })
            }
        };

        SerializableChatMessage {
            role: msg.role,
            content: msg.content,
            extra,
        }
    }
}

impl From<SerializableChatMessage> for ChatMessage {
    fn from(s_msg: SerializableChatMessage) -> Self {
        let message_type = match s_msg.extra {
            None => MessageType::Text,
            Some(SerializableMessageExtra::Image { mime, bytes }) => {
                MessageType::Image((mime, bytes))
            }
            Some(SerializableMessageExtra::Pdf { bytes }) => MessageType::Pdf(bytes),
            Some(SerializableMessageExtra::ImageUrl { url }) => MessageType::ImageURL(url),
            Some(SerializableMessageExtra::ToolUse { calls }) => MessageType::ToolUse(calls),
            Some(SerializableMessageExtra::ToolResult { results }) => {
                MessageType::ToolResult(results)
            }
        };

        ChatMessage {
            role: s_msg.role,
            content: s_msg.content,
            message_type,
        }
    }
}

/// Serializes a vector of ChatMessages to a JSON string.
///
/// This function converts the internal `ChatMessage` representation into a
/// serializable format, including base64-encoding for binary data,
/// and returns it as a JSON string.
///
/// # Example
/// ```
/// # use llm::chat::{serialize_messages, ChatMessage, ChatRole, MessageType};
/// let messages = vec![ChatMessage::user().content("Hello, world!").build()];
/// let json_string = serialize_messages(&messages).unwrap();
/// assert_eq!(json_string, r#"[{"role":"User","content":"Hello, world!"}]"#);
/// ```
pub fn serialize_messages(messages: &[ChatMessage]) -> Result<String, serde_json::Error> {
    let serializable_msgs: Vec<SerializableChatMessage> =
        messages.iter().cloned().map(Into::into).collect();
    serde_json::to_string_pretty(&serializable_msgs)
}

/// Deserializes a JSON string into a vector of ChatMessages.
///
/// This function parses a JSON string, expecting it to match the format
/// produced by `serialize_messages`, and converts it back into the
/// library's internal `ChatMessage` representation.
pub fn deserialize_messages(json_str: &str) -> Result<Vec<ChatMessage>, serde_json::Error> {
    let serializable_msgs: Vec<SerializableChatMessage> = serde_json::from_str(json_str)?;
    let messages: Vec<ChatMessage> = serializable_msgs.into_iter().map(Into::into).collect();
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FunctionCall, ToolCall};

    #[test]
    fn test_chat_message_serialization_and_deserialization() {
        let messages = vec![
            ChatMessage::user().content("Hello!").build(),
            ChatMessage::assistant()
                .content("I have a function call for you.")
                .tool_use(vec![ToolCall {
                    id: "call_123".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: r#"{"location": "Tokyo"}"#.to_string(),
                    },
                }])
                .build(),
            ChatMessage::user()
                .content("Here is a picture of my cat.")
                .image(ImageMime::JPEG, vec![255, 216, 255, 224]) // Fake JPEG header
                .build(),
        ];

        // Serialize
        let json_string = serialize_messages(&messages).unwrap();
        println!("Serialized JSON:\n{json_string}");

        // Deserialize
        let deserialized_messages = deserialize_messages(&json_string).unwrap();

        // Verify
        assert_eq!(messages.len(), deserialized_messages.len());

        // Message 1: Simple text
        assert_eq!(messages[0].role, deserialized_messages[0].role);
        assert_eq!(messages[0].content, deserialized_messages[0].content);
        assert_eq!(
            messages[0].message_type,
            deserialized_messages[0].message_type
        );

        // Message 2: Tool Use
        assert_eq!(messages[1].role, deserialized_messages[1].role);
        assert_eq!(messages[1].content, deserialized_messages[1].content);
        assert_eq!(
            messages[1].message_type,
            deserialized_messages[1].message_type
        );
        if let MessageType::ToolUse(calls) = &deserialized_messages[1].message_type {
            assert_eq!(calls[0].function.name, "get_weather");
        } else {
            panic!("Expected ToolUse message type");
        }

        // Message 3: Image
        assert_eq!(messages[2].role, deserialized_messages[2].role);
        assert_eq!(messages[2].content, deserialized_messages[2].content);
        assert_eq!(
            messages[2].message_type,
            deserialized_messages[2].message_type
        );
        if let MessageType::Image((mime, bytes)) = &deserialized_messages[2].message_type {
            assert_eq!(*mime, ImageMime::JPEG);
            assert_eq!(*bytes, vec![255, 216, 255, 224]);
        } else {
            panic!("Expected Image message type");
        }
    }
}
