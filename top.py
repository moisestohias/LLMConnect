"""
HTTP LLM-Providers Client using only Python standard library.
"""


import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator, Union, Iterator

from base import (
    SyncHTTPClient, AsyncHTTPClient,
    create_sync_client, create_async_client,
    ConnectionPool, RetryConfig, BaseMiddleware,
    AuthenticationMiddleware, UserAgentMiddleware, LoggingMiddleware,
    HTTPResponse, api_key, user_agent
)

# Core API Logic
class APIExecutor:
    """Core API logic shared between sync and async clients."""

    def __init__(self, api_key: str, endpoint: str = "chat/completions", 
                 base_url: str = "https://api.cerebras.ai/v1",
                 model: str = "llama-3.3-70b",
                 temperature: float = 0.7,
                 max_completion_tokens: int = 100,
                 timeout: float = 30.0):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/').lstrip('/')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        # LLM-specific parameters
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.messages = []

    def set_parameters(self, model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_completion_tokens: Optional[int] = None):
        """Update LLM parameters."""
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_completion_tokens is not None:
            self.max_completion_tokens = max_completion_tokens

    def clear_messages(self):
        """Clear the message history."""
        self.messages = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def prepare_request_data(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Prepare the request data for the API call."""
        # Add user message to history
        self.add_message("user", prompt)

        data = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens
        }

        if stream:
            data["stream"] = True

        return data

    def get_request_config(self, data: Dict[str, Any],
                          headers: Optional[Dict[str, str]] = None) -> tuple:
        """Get the request configuration (URL, headers, body)."""
        url = f"{self.base_url}/{self.endpoint}"
        body = json.dumps(data).encode('utf-8')

        request_headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.api_key}"
        }

        if headers:
            request_headers.update(headers)

        return url, request_headers, body

    def process_non_streaming_response(self, response: HTTPResponse) -> str:
        """Process a non-streaming chat response."""
        result = json.loads(response.body.decode('utf-8'))
        assistant_message = result["choices"][0]["message"]["content"]

        # Add assistant's response to history
        self.add_message("assistant", assistant_message)

        return assistant_message

    def parse_streaming_chunk(self, chunk: bytes) -> Optional[str]:
        """Parse a streaming chunk and extract content."""
        chunk_str = chunk.decode('utf-8', errors='ignore')
        
        # Handle Server-Sent Events format
        lines = chunk_str.strip().split('\n')
        content_parts = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("data: "):
                chunk_data = line[6:].strip()
                
                if chunk_data == "[DONE]":
                    return None  # End of stream
                
                if chunk_data:  # Skip empty data lines
                    try:
                        parsed = json.loads(chunk_data)
                        if "choices" in parsed and len(parsed["choices"]) > 0:
                            delta = parsed["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    except json.JSONDecodeError:
                        # Log but don't fail on parse errors
                        logger.debug(f"Failed to parse SSE data: {chunk_data}")
                        continue
        
        # Return concatenated content from all events in this chunk
        return "".join(content_parts) if content_parts else ""

# Synchronous API Client
class SyncAPIClient:
    """Synchronous LLM API client."""

    def __init__(self, api_key: str, endpoint: str, base_url: str,
                 model: str = "llama-3.3-70b",
                 temperature: float = 0.7,
                 max_completion_tokens: int = 100,
                 timeout: float = 30.0,
                 http_client: Optional[SyncHTTPClient] = None,
                 middleware: Optional[List[BaseMiddleware]] = None,
                 connection_pool: Optional[ConnectionPool] = None,
                 retry_config: Optional[RetryConfig] = None):

        self._executor = APIExecutor(
            api_key, endpoint, base_url, model, temperature, max_completion_tokens, timeout
        )

        # Use provided client or create a new one
        if http_client:
            self._http_client = http_client
            self._owns_client = False
        else:
            # Create default middleware if none provided
            if middleware is None:
                middleware = [
                    AuthenticationMiddleware(api_key),
                    UserAgentMiddleware(user_agent),
                    LoggingMiddleware()
                ]

            self._http_client = SyncHTTPClient(connection_pool, retry_config, middleware)
            self._owns_client = True

    @property
    def model(self) -> str:
        return self._executor.model

    @property
    def temperature(self) -> float:
        return self._executor.temperature

    @property
    def max_completion_tokens(self) -> int:
        return self._executor.max_completion_tokens

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._executor.messages

    def set_parameters(self, model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_completion_tokens: Optional[int] = None):
        """Update LLM parameters."""
        self._executor.set_parameters(model, temperature, max_completion_tokens)

    def clear_messages(self):
        """Clear the message history."""
        self._executor.clear_messages()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self._executor.add_message(role, content)

    def chat(self, prompt: str, stream: bool = False) -> Union[str, Iterator[str]]:
        """Send a chat message and get a response."""
        data = self._executor.prepare_request_data(prompt, stream)

        if stream:
            return self._stream_chat(data)
        else:
            url, headers, body = self._executor.get_request_config(data)
            response = self._http_client.request('POST', url, headers=headers,
                                               body=body, timeout=self._executor.timeout)
            return self._executor.process_non_streaming_response(response)

    def _stream_chat(self, data: Dict[str, Any]) -> Iterator[str]:
        """Handle streaming chat responses."""
        url, headers, body = self._executor.get_request_config(data)
        headers['Accept'] = 'text/event-stream'

        full_response = []

        for chunk in self._http_client.stream_request('POST', url, headers=headers,
                                                    body=body, timeout=self._executor.timeout):
            content = self._executor.parse_streaming_chunk(chunk)

            if content is None:  # End of stream
                break
            elif content:  # Non-empty content
                full_response.append(content)
                yield content

        # Add complete response to history
        if full_response:
            self._executor.add_message("assistant", "".join(full_response))

    def close(self):
        """Close the client and cleanup resources."""
        if self._owns_client:
            self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Asynchronous API Client
class AsyncAPIClient:
    """Asynchronous LLM API client."""

    def __init__(self, api_key: str, endpoint: str, base_url: str,
                model: str = "llama-3.3-70b",
                temperature: float = 0.7,
                max_completion_tokens: int = 100,
                timeout: float = 30.0,
                http_client: Optional[AsyncHTTPClient] = None,
                middleware: Optional[List[BaseMiddleware]] = None,
                connection_pool: Optional[ConnectionPool] = None,
                retry_config: Optional[RetryConfig] = None):

        self._executor = APIExecutor(
            api_key, endpoint, base_url, model, temperature, max_completion_tokens, timeout
        )

        # Use provided client or create a new one
        if http_client:
            self._http_client = http_client
            self._owns_client = False
        else:
            # Create default middleware if none provided
            if middleware is None:
                middleware = [
                    AuthenticationMiddleware(api_key),
                    UserAgentMiddleware(user_agent),
                    LoggingMiddleware()
                ]

            self._http_client = AsyncHTTPClient(connection_pool, retry_config, middleware)
            self._owns_client = True

    @property
    def model(self) -> str:
        return self._executor.model

    @property
    def temperature(self) -> float:
        return self._executor.temperature

    @property
    def max_completion_tokens(self) -> int:
        return self._executor.max_completion_tokens

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._executor.messages

    def set_parameters(self, model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_completion_tokens: Optional[int] = None):
        """Update LLM parameters."""
        self._executor.set_parameters(model, temperature, max_completion_tokens)

    def clear_messages(self):
        """Clear the message history."""
        self._executor.clear_messages()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self._executor.add_message(role, content)

    async def chat(self, prompt: str, stream: bool = False) -> Union[str, AsyncIterator[str]]:
        """Send a chat message and get a response."""
        data = self._executor.prepare_request_data(prompt, stream)

        if stream:
            # Don't await here - return the async generator directly
            return self._stream_chat(data)
        else:
            url, headers, body = self._executor.get_request_config(data)
            response = await self._http_client.request('POST', url, headers=headers,
                                                     body=body, timeout=self._executor.timeout)
            return self._executor.process_non_streaming_response(response)

    async def _stream_chat(self, data: Dict[str, Any]) -> AsyncIterator[str]:
        """Handle streaming chat responses."""
        url, headers, body = self._executor.get_request_config(data)
        headers['Accept'] = 'text/event-stream'

        full_response = []

        async for chunk in self._http_client.stream_request('POST', url, headers=headers,
                                                          body=body, timeout=self._executor.timeout):
            content = self._executor.parse_streaming_chunk(chunk)

            if content is None:  # End of stream
                break
            elif content:  # Non-empty content
                full_response.append(content)
                yield content

        # Add complete response to history
        if full_response:
            self._executor.add_message("assistant", "".join(full_response))

    async def close(self):
        """Close the client and cleanup resources."""
        if self._owns_client:
            await self._http_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Factory Functions
def create_sync_api_client(api_key: str, endpoint: str, base_url: str,
                          model: str = "llama-3.3-70b",
                          temperature: float = 0.7,
                          max_completion_tokens: int = 100,
                          timeout: float = 30.0,
                          **kwargs) -> SyncAPIClient:
    """Create a configured synchronous API client."""
    return SyncAPIClient(
        api_key=api_key,
        endpoint=endpoint,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        **kwargs
    )

def create_async_api_client(api_key: str, endpoint: str, base_url: str,
                           model: str = "llama-3.3-70b",
                           temperature: float = 0.7,
                           max_completion_tokens: int = 100,
                           timeout: float = 30.0,
                           **kwargs) -> AsyncAPIClient:
    """Create a configured asynchronous API client."""
    return AsyncAPIClient(
        api_key=api_key,
        endpoint=endpoint,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        **kwargs
    )

# Convenience functions for common LLM providers
def create_cerebras_sync_client(api_key: Optional[str] = None,
                               model: str = "llama-3.3-70b",
                               **kwargs) -> SyncAPIClient:
    """Create a sync client for Cerebras API."""
    return create_sync_api_client(
        api_key=api_key or globals().get('api_key', ''),
        endpoint="chat/completions",
        base_url="https://api.cerebras.ai/v1",
        model=model,
        **kwargs
    )

def create_cerebras_async_client(api_key: Optional[str] = None,
                                model: str = "llama-3.3-70b",
                                **kwargs) -> AsyncAPIClient:
    """Create an async client for Cerebras API."""
    return create_async_api_client(
        api_key=api_key or globals().get('api_key', ''),
        endpoint="chat/completions",
        base_url="https://api.cerebras.ai/v1",
        model=model,
        **kwargs
    )

# Backward compatibility aliases
APIClient = AsyncAPIClient
BaseAPIClient = AsyncAPIClient

# Example usage functions
def sync_example():
    """Example of using the synchronous API client."""
    print("=== Synchronous API Client Example ===")

    with create_cerebras_sync_client() as client:
        try:
            # Simple chat
            response = client.chat("Say hello and nothing else")
            print(f"Assistant: {response}")

            # Continue conversation
            response = client.chat("Now say it in Spanish")
            print(f"Assistant: {response}")

            # Streaming chat
            print("Streaming response: ", end="")
            for chunk in client.chat("Count from 1 to 5", stream=True):
                print(chunk, end="", flush=True)
            print()

            # print(f"Conversation has {len(client.messages)} messages")

        except Exception as e:
            print(f"Error: {e}")

async def async_example():
    """Example of using the asynchronous API client."""
    print("\n=== Asynchronous API Client Example ===")

    async with create_cerebras_async_client() as client:
        try:
            # Simple chat
            response = await client.chat("Say hello and nothing else")
            print(f"Assistant: {response}")

            # Continue conversation
            response = await client.chat("Now say it in French")
            print(f"Assistant: {response}")

            # Streaming chat - don't await the chat call for streaming
            print("Streaming response: ", end="")
            stream = await client.chat("Count from 1 to 3", stream=True)  # No await here
            async for chunk in stream:  # Iterate over the async generator
                print(chunk, end="", flush=True)
            print()

            print(f"Conversation has {len(client.messages)} messages")

        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run examples demonstrating both sync and async usage."""
    # Test sync client
    # sync_example()

    # Test async client
    asyncio.run(async_example())

if __name__ == "__main__":
    main()
