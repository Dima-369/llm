use llm::completion::CompletionProvider;
use llm::models::Models;
use llm::prelude::*;
use llm::secret_store::SecretStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Models::Together;
    let secrets = SecretStore::new();
    let provider = model.to_provider(&secrets).await?;

    let request = CompletionRequest {
        prompt: "What is the capital of France?".to_string(),
        ..Default::default()
    };

    let response = provider.complete(&request).await?;

    println!("Response: {}", response.text);

    Ok(())
}
