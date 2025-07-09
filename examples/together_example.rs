use llm::completion::CompletionRequest;
use llm::models::Models;
use llm::secret_store::SecretStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Models::Together;
    let secrets = SecretStore::new()?;
    let provider = model.to_provider(&secrets).await?;
    let request = CompletionRequest::new("What is the capital of France?");
    let response = provider.complete(&request).await?;
    println!("Response: {}", response.text);
    Ok(())
}
