#![allow(unused)]

//! This example demonstrates how to use the Copilot provider.
//!
//! To run this example, you need to have a GitHub token with Copilot access.
//! You can either set the `COPILOT_GITHUB_TOKEN` environment variable, or the
//! application will guide you through an interactive authentication process.
//!
//! Usage:
//! `cargo run --example copilot_example --features copilot`

use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::ChatMessage;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // It's recommended to set the log level to debug to see the authentication flow.
    // You can do this by running `RUST_LOG=debug cargo run ...`
    // The `env_logger` crate is a dev-dependency, so it's not available in the example by default.
    // To enable it, you would need to add it to the [dev-dependencies] section of Cargo.toml
    // and uncomment the line below.
    // let _ = env_logger::try_init();

    let user_message = "say hi";

    // Create a new LLM provider with the Copilot backend.
    // If the `COPILOT_GITHUB_TOKEN` environment variable is not set,
    // an interactive device authentication flow will be initiated.
    let temp_dir = tempdir()?;
    let provider = LLMBuilder::new()
        .backend(LLMBackend::Copilot)
        .github_copilot_token_directory(
            temp_dir
                .path()
                .to_str()
                .ok_or("Failed to convert path to string")?,
        )
        .build()?;

    let messages = vec![ChatMessage::user()
        .content(user_message.to_string())
        .build()];

    println!("Sending message to Copilot: \"{user_message}\"");

    // Send the chat message and await the response.
    let response = provider.chat(&messages).await?;

    println!("\nCopilot's response:\n---\n{response}\n---");

    Ok(())
}
