// Import required modules from the LLM library for Cohere integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Cohere API key from environment variable or use test key as fallback
    let api_key = std::env::var("COHERE_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Cohere) // Use Cohere as the LLM provider
        .proxy_url("http://localhost:9090")
        .api_key(api_key) // Set the API key
        .model("command-r-plus") // Use command-r-plus model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (Cohere)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user()
            .content("Tell me that you love cats")
            .build(),
        ChatMessage::assistant()
            .content("I am an assistant, I cannot love cats but I can love dogs")
            .build(),
        ChatMessage::user()
            .content("Tell me that you love dogs in 2000 chars")
            .build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }

    Ok(())
}