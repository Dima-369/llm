// X.AI streaming chat example demonstrating real-time token generation
use futures::StreamExt;
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get X.AI API key from environment variable or use test key as fallback
    let api_key = std::env::var("XAI_API_KEY").unwrap_or("xai-TESTKEY".into());

    // Initialize and configure the LLM client with streaming enabled
    let llm = LLMBuilder::new()
        .backend(LLMBackend::XAI)
        .api_key(api_key)
        .model("grok-2-latest")
        .max_tokens(1000)
        .temperature(0.7)
        .stream(true) // Enable streaming responses
        .build()
        .expect("Failed to build LLM (X.AI)");

    // Prepare conversation with a prompt that will generate a longer response
    let messages = vec![ChatMessage::user()
        .content(
            "Write a long story about a robot learning to paint. Make it creative and engaging.",
        )
        .build()];

    println!("Starting streaming chat with X.AI...\n");

    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();

            while let Some(Ok(token)) = stream.next().await {
                handle.write_all(token.as_bytes()).unwrap();
                handle.flush().unwrap();
            }
            println!("\n\nStreaming completed.");
        }
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
