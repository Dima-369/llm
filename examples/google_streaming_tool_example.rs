// Google streaming chat example demonstrating real-time token generation with a tool that opens a file
use futures::StreamExt;
use llm::{
    builder::{LLMBackend, LLMBuilder, FunctionBuilder, ParamBuilder},
    chat::ChatMessage,
};
use std::io::{self, Write};
use std::fs::File;
use std::io::Read;
use llm::chat::ParameterProperty;

fn open_files_tool() -> FunctionBuilder {
    FunctionBuilder::new("open_files")
        .description("Open one or more files in the workspace.")
        .param(
            ParamBuilder::new("file_paths")
                .type_of("array")
                .items(ParameterProperty {
                    property_type: "string".to_string(),
                    description: "A file path".to_string(),
                    items: None,
                    enum_list: None,
                })
                .description("A list of file paths to open."),
        )
        .required(vec!["file_paths".to_string()])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable or use test key as fallback
    let api_key = std::env::var("GOOGLE_API_KEY=").unwrap_or("TESTKEY".into());

    // Initialize and configure the LLM client with streaming enabled and the open_files tool
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Google)
        .api_key(api_key)
        .model("gemini-2.0-flash")
        .proxy_url("http://localhost:9090")
        .max_tokens(1000)
        .temperature(0.7)
        .stream(true) // Enable streaming responses
        .function(open_files_tool())
        .build()
        .expect("Failed to build LLM (Google)");

    // Prepare conversation with a prompt that will generate a longer response
    let messages = vec![ChatMessage::user()
        .content(
            "open readme",
        )
        .build()];

    println!("Starting streaming chat with Google...\n");

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