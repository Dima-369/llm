use crate::{builder::LLMBackend, error::LLMError, secret_store::SecretStore};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum Models {
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    Deepseek,
    Groq,
    Ollama,
    Phind,
    Xai,
    Copilot,
    AzureOpenAI,
    ElevenLabs,
    Together,
}

impl Models {
    pub async fn to_provider(
        &self,
        secrets: &SecretStore,
    ) -> Result<Box<dyn crate::LLMProvider>, LLMError> {
        match self {
            Models::OpenAI => Ok(Box::new(crate::backends::openai::OpenAI::new(
                secrets.get("OPENAI_API_KEY").map(|s| s.to_string()),
                secrets.get("OPENAI_PROXY_URL").map(|s| s.to_string()),
                secrets.get("OPENAI_BASE_URL").map(|s| s.to_string()),
                secrets.get("OPENAI_BASE_URL_ABSOLUTE").is_some(),
                secrets.get("OPENAI_MODEL").map(|s| s.to_string()),
                secrets
                    .get("OPENAI_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("OPENAI_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("OPENAI_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("OPENAI_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("OPENAI_STREAM").is_some()),
                secrets.get("OPENAI_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("OPENAI_TOP_K").and_then(|s| s.parse().ok()),
                secrets
                    .get("OPENAI_EMBEDDING_ENCODING_FORMAT")
                    .map(|s| s.to_string()),
                secrets
                    .get("OPENAI_EMBEDDING_DIMENSIONS")
                    .and_then(|s| s.parse().ok()),
                None,
                None,
                secrets
                    .get("OPENAI_REASONING_EFFORT")
                    .map(|s| s.to_string()),
                None,
                secrets.get("OPENAI_VOICE").map(|s| s.to_string()),
                Some(secrets.get("OPENAI_ENABLE_WEB_SEARCH").is_some()),
                secrets
                    .get("OPENAI_WEB_SEARCH_CONTEXT_SIZE")
                    .map(|s| s.to_string()),
                secrets
                    .get("OPENAI_WEB_SEARCH_USER_LOCATION_TYPE")
                    .map(|s| s.to_string()),
                secrets
                    .get("OPENAI_WEB_SEARCH_USER_LOCATION_APPROXIMATE_COUNTRY")
                    .map(|s| s.to_string()),
                secrets
                    .get("OPENAI_WEB_SEARCH_USER_LOCATION_APPROXIMATE_CITY")
                    .map(|s| s.to_string()),
                secrets
                    .get("OPENAI_WEB_SEARCH_USER_LOCATION_APPROXIMATE_REGION")
                    .map(|s| s.to_string()),
            ))),
            Models::Anthropic => Ok(Box::new(crate::backends::anthropic::Anthropic::new(
                secrets.get("ANTHROPIC_API_KEY").map(|s| s.to_string()),
                secrets.get("ANTHROPIC_PROXY_URL").map(|s| s.to_string()),
                secrets.get("ANTHROPIC_MODEL").map(|s| s.to_string()),
                secrets
                    .get("ANTHROPIC_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("ANTHROPIC_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("ANTHROPIC_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("ANTHROPIC_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("ANTHROPIC_STREAM").is_some()),
                secrets.get("ANTHROPIC_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("ANTHROPIC_TOP_K").and_then(|s| s.parse().ok()),
                None,
                None,
                secrets
                    .get("ANTHROPIC_REASONING_EFFORT")
                    .and_then(|s| s.parse::<bool>().ok()),
                secrets
                    .get("ANTHROPIC_THINKING_BUDGET_TOKENS")
                    .and_then(|s| s.parse().ok()),
            ))),
            Models::Google => Ok(Box::new(crate::backends::google::Google::new(
                secrets.get("GOOGLE_API_KEY").map(|s| s.to_string()),
                secrets.get("GOOGLE_PROXY_URL").map(|s| s.to_string()),
                secrets.get("GOOGLE_MODEL").map(|s| s.to_string()),
                secrets
                    .get("GOOGLE_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("GOOGLE_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("GOOGLE_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("GOOGLE_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("GOOGLE_STREAM").is_some()),
                secrets.get("GOOGLE_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("GOOGLE_TOP_K").and_then(|s| s.parse().ok()),
                None,
                None,
            ))),
            Models::Cohere => Ok(Box::new(crate::backends::cohere::Cohere::new(
                secrets.get("COHERE_API_KEY").map(|s| s.to_string()),
                secrets.get("COHERE_PROXY_URL").map(|s| s.to_string()),
                secrets.get("COHERE_BASE_URL").map(|s| s.to_string()),
                secrets.get("COHERE_BASE_URL_ABSOLUTE").is_some(),
                secrets.get("COHERE_MODEL").map(|s| s.to_string()),
                secrets
                    .get("COHERE_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("COHERE_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("COHERE_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("COHERE_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("COHERE_STREAM").is_some()),
                secrets.get("COHERE_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("COHERE_TOP_K").and_then(|s| s.parse().ok()),
                None,
                None,
                None,
                None,
                secrets
                    .get("COHERE_REASONING_EFFORT")
                    .map(|s| s.to_string()),
                None,
            ))),
            Models::Deepseek => Ok(Box::new(crate::backends::deepseek::DeepSeek::new(
                secrets.get("DEEPSEEK_API_KEY").map(|s| s.to_string()),
                secrets.get("DEEPSEEK_PROXY_URL").map(|s| s.to_string()),
                secrets.get("DEEPSEEK_MODEL").map(|s| s.to_string()),
                secrets
                    .get("DEEPSEEK_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("DEEPSEEK_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("DEEPSEEK_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("DEEPSEEK_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("DEEPSEEK_STREAM").is_some()),
            ))),
            Models::Groq => Ok(Box::new(crate::backends::groq::Groq::new(
                secrets.get("GROQ_API_KEY").map(|s| s.to_string()),
                secrets.get("GROQ_PROXY_URL").map(|s| s.to_string()),
                secrets.get("GROQ_MODEL").map(|s| s.to_string()),
                secrets.get("GROQ_MAX_TOKENS").and_then(|s| s.parse().ok()),
                secrets.get("GROQ_TEMPERATURE").and_then(|s| s.parse().ok()),
                secrets
                    .get("GROQ_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("GROQ_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("GROQ_STREAM").is_some()),
                secrets.get("GROQ_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("GROQ_TOP_K").and_then(|s| s.parse().ok()),
            ))),
            Models::Ollama => Ok(Box::new(crate::backends::ollama::Ollama::new(
                secrets
                    .get("OLLAMA_BASE_URL")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                secrets.get("OLLAMA_API_KEY").map(|s| s.to_string()),
                secrets.get("OLLAMA_PROXY_URL").map(|s| s.to_string()),
                secrets.get("OLLAMA_MODEL").map(|s| s.to_string()),
                secrets
                    .get("OLLAMA_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("OLLAMA_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("OLLAMA_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("OLLAMA_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("OLLAMA_STREAM").is_some()),
                secrets.get("OLLAMA_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("OLLAMA_TOP_K").and_then(|s| s.parse().ok()),
                None,
                None,
            ))),
            Models::Phind => Ok(Box::new(crate::backends::phind::Phind::new(
                secrets.get("PHIND_MODEL").map(|s| s.to_string()),
                secrets.get("PHIND_PROXY_URL").map(|s| s.to_string()),
                secrets.get("PHIND_MAX_TOKENS").and_then(|s| s.parse().ok()),
                secrets
                    .get("PHIND_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("PHIND_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("PHIND_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("PHIND_STREAM").is_some()),
                secrets.get("PHIND_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("PHIND_TOP_K").and_then(|s| s.parse().ok()),
            ))),
            Models::Xai => Ok(Box::new(crate::backends::xai::XAI::new(
                secrets.get("XAI_API_KEY").map(|s| s.to_string()),
                secrets.get("XAI_PROXY_URL").map(|s| s.to_string()),
                secrets.get("XAI_MODEL").map(|s| s.to_string()),
                secrets.get("XAI_MAX_TOKENS").and_then(|s| s.parse().ok()),
                secrets.get("XAI_TEMPERATURE").and_then(|s| s.parse().ok()),
                secrets
                    .get("XAI_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("XAI_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("XAI_STREAM").is_some()),
                secrets.get("XAI_TOP_P").and_then(|s| s.parse().ok()),
                secrets.get("XAI_TOP_K").and_then(|s| s.parse().ok()),
                secrets
                    .get("XAI_EMBEDDING_ENCODING_FORMAT")
                    .map(|s| s.to_string()),
                secrets
                    .get("XAI_EMBEDDING_DIMENSIONS")
                    .and_then(|s| s.parse().ok()),
                None, // json_schema
                secrets.get("XAI_WEB_SEARCH_MODE").map(|s| s.to_string()),
                secrets
                    .get("XAI_WEB_SEARCH_SOURCE_TYPE")
                    .map(|s| s.to_string()),
                secrets
                    .get("XAI_WEB_SEARCH_EXCLUDED_WEBSITES")
                    .map(|s| s.split(',').map(|s| s.to_string()).collect()),
                secrets
                    .get("XAI_WEB_SEARCH_MAX_RESULTS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("XAI_WEB_SEARCH_FROM_DATE")
                    .map(|s| s.to_string()),
                secrets.get("XAI_WEB_SEARCH_TO_DATE").map(|s| s.to_string()),
            ))),
            Models::Copilot => Ok(Box::new(crate::backends::copilot::Copilot::new(
                secrets.get("COPILOT_API_KEY").map(|s| s.to_string()),
                secrets.get("COPILOT_PROXY_URL").map(|s| s.to_string()),
                secrets.get("COPILOT_MODEL").map(|s| s.to_string()),
                secrets
                    .get("COPILOT_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("COPILOT_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("COPILOT_SYSTEM").map(|s| s.to_string()),
                None,
                None,
                std::path::PathBuf::from("/tmp"),
            )?)),
            Models::AzureOpenAI => Ok(Box::new(crate::backends::azure_openai::AzureOpenAI::new(
                secrets.get("AZURE_OPENAI_API_KEY").map(|s| s.to_string()),
                secrets.get("AZURE_OPENAI_PROXY_URL").map(|s| s.to_string()),
                secrets
                    .get("AZURE_OPENAI_API_VERSION")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                secrets
                    .get("AZURE_OPENAI_DEPLOYMENT_ID")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                secrets
                    .get("AZURE_OPENAI_ENDPOINT")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                secrets.get("AZURE_OPENAI_BASE_URL_ABSOLUTE").is_some(),
                secrets.get("AZURE_OPENAI_MODEL").map(|s| s.to_string()),
                secrets
                    .get("AZURE_OPENAI_MAX_TOKENS")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("AZURE_OPENAI_TEMPERATURE")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("AZURE_OPENAI_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("AZURE_OPENAI_SYSTEM").map(|s| s.to_string()),
                Some(secrets.get("AZURE_OPENAI_STREAM").is_some()),
                secrets
                    .get("AZURE_OPENAI_TOP_P")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("AZURE_OPENAI_TOP_K")
                    .and_then(|s| s.parse().ok()),
                secrets
                    .get("AZURE_OPENAI_EMBEDDING_ENCODING_FORMAT")
                    .map(|s| s.to_string()),
                secrets
                    .get("AZURE_OPENAI_EMBEDDING_DIMENSIONS")
                    .and_then(|s| s.parse().ok()),
                None,
                None,
                secrets
                    .get("AZURE_OPENAI_REASONING_EFFORT")
                    .map(|s| s.to_string()),
                None,
            ))),
            Models::ElevenLabs => Ok(Box::new(crate::backends::elevenlabs::ElevenLabs::new(
                secrets.get("ELEVENLABS_API_KEY").map(|s| s.to_string()),
                secrets.get("ELEVENLABS_PROXY_URL").map(|s| s.to_string()),
                secrets
                    .get("ELEVENLABS_MODEL")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                secrets
                    .get("ELEVENLABS_BASE_URL")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                secrets
                    .get("ELEVENLABS_TIMEOUT_SECONDS")
                    .and_then(|s| s.parse().ok()),
                secrets.get("ELEVENLABS_VOICE").map(|s| s.to_string()),
            ))),
            Models::Together => Ok(Box::new(crate::backends::together::Together::new(
                secrets.get("TOGETHER_PROXY_URL").map(|s| s.to_string()),
                secrets,
                None,
                None,
            ))),
        }
    }
}

impl std::fmt::Display for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

pub trait ModelListResponse {
    fn get_models(&self) -> Vec<String>;
    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>>;
    fn get_backend(&self) -> LLMBackend;
}

pub trait ModelListRawEntry: Debug {
    fn get_id(&self) -> String;
    fn get_created_at(&self) -> DateTime<Utc>;
    fn get_raw(&self) -> serde_json::Value;
}

#[derive(Debug, Clone, Default)]
pub struct ModelListRequest {
    pub filter: Option<String>,
}

/// Trait for providers that support listing and retrieving model information.
#[async_trait]
pub trait ModelsProvider {
    /// Asynchronously retrieves the list of available models ID's from the provider.
    ///
    /// # Arguments
    ///
    /// * `_request` - Optional filter by model ID
    ///
    /// # Returns
    ///
    /// List of model ID's or error
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "List Models not supported".to_string(),
        ))
    }
}
