# HTTP LLM-Providers API Client

A Python HTTP client implementation for interacting with Large Language Model (LLM) providers, featuring connection pooling, retry logic, middleware support, and both synchronous and asynchronous interfaces.

## Project Overview

This project provides a standardized interface for communicating with various LLM providers through their HTTP APIs. It implements robust error handling, automatic retries with exponential backoff, connection pooling for efficiency, and a middleware system for extensibility. The client supports both synchronous and asynchronous programming patterns and is compatible with OpenAI-style chat completion endpoints.

Key features:
- Support for multiple LLM providers (Cerebras, Groq, OpenRouter,..)
- Connection pooling and reuse
- Configurable retry logic with exponential backoff
- Middleware system for authentication, logging, and metrics
- Both synchronous and asynchronous interfaces
- Streaming and non-streaming response handling

## Main Components

### Core Components
- **HTTPRequest/HTTPResponse**: Data models representing HTTP requests and responses
- **ConnectionPool**: Thread-safe connection management with reuse
- **RetryConfig**: Configurable retry behavior with exponential backoff
- **RequestExecutor**: Core request execution logic shared between sync/async clients

### Middleware System
- **AuthenticationMiddleware**: Adds API key authentication headers
- **LoggingMiddleware**: Logs requests and responses
- **UserAgentMiddleware**: Adds User-Agent headers
- **MetricsMiddleware**: Collects request metrics and statistics

### Client Implementations
- **SyncAPIClient** and **AsyncAPIClient**: Sync/Async LLM API client with chat-specific functionality

### Factory System
- **APIClientFactory**: Creates provider-specific clients with validated configuration
- **ProviderConfig**: Provider-specific configuration management

## Usage Instructions

### Basic Usage

```python
from llm_client import create_groq_sync_client

# Create a client
client = create_groq_sync_client(api_key="your-api-key")

# Send a chat message
response = client.chat("Hello, how are you?")
print(response)

# Stream responses
for chunk in client.chat("Tell me a story", stream=True):
    print(chunk, end="", flush=True)
```

### Advanced Usage

```python
from llm_client import SyncAPIClient, AuthenticationMiddleware, LoggingMiddleware

# Create a custom client
client = SyncAPIClient(
    api_key="your-api-key",
    base_url="https://api.example.com",
    model="example-model",
    middleware=[
        AuthenticationMiddleware("your-api-key"),
        LoggingMiddleware()
    ]
)

messages = []
def append_user_prompt(prompt): messages.append({'role': 'user', 'content': prompt})
def append_assistant_prompt(prompt): messages.append({'role': 'assistant', 'content': prompt})

user_prompt_1 = "say hello and nothing else"
user_prompt_2 = "now say it Spanish"

## Auto-Conversation Management Using Chat
resp = client.chat(user_prompt_1)
print(resp)
resp = client.chat(user_prompt_2)
print(resp)
print(client.messages)


## Manual-Conversation Management Using Send
append_user_prompt(user_prompt_1)
resp = client.send(messages)
append_assistant_prompt(resp)
print(resp)

append_user_prompt(user_prompt_2)
resp = client.send(messages)
append_assistant_prompt(resp)
print(resp)
```

### Asynchronous Usage

```python
from llm_client import create_openrouter_async_client
import asyncio

async def main():
    client = await create_openrouter_async_client(api_key="your-api-key")
    
    async for chunk in client.chat("Write a poem", stream=True):
        print(chunk, end="", flush=True)
    
    await client.close()

asyncio.run(main())
```

## Configuration

### Environment Variables

Set API keys as environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export CEREBRAS_API_KEY="your-cerebras-api-key"
```

### Provider Configuration

Provider configurations are stored in `provider_configs.json`

### Custom Configuration

```python
from llm_client import SyncAPIClient, RetryConfig, ConnectionPool

# Custom connection pool
pool = ConnectionPool(max_connections_per_host=5)

# Custom retry configuration
retry_config = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    backoff_factor=1.5
)

client = SyncAPIClient(
    api_key="your-api-key",
    base_url="https://api.example.com",
    model="custom-model",
    connection_pool=pool,
    retry_config=retry_config
)
```

### Middleware Configuration

```python
from llm_client import MetricsMiddleware

metrics = MetricsMiddleware()

client = SyncAPIClient(
    api_key="your-api-key",
    base_url="https://api.example.com",
    model="example-model",
    middleware=[metrics]
)

# After making requests
stats = metrics.get_stats()
print(f"Success rate: {(1 - stats['error_rate']) * 100:.1f}%")
```