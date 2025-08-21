"""
HTTP LLM-Providers API Client using only Python standard library.
This was the original version. I've added send & post to both SyncAPIClient & AsyncAPIClient
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator, Union, Iterator

from base import SyncHTTPClient, AsyncHTTPClient, ConnectionPool, RetryConfig
from middlewares import AuthenticationMiddleware, UserAgentMiddleware, LoggingMiddleware, HTTPResponse

from utils import validate_messages_format
user_agent: str = "APIClient/1.0.0"


# Core API Logic
class APIExecutor:
    """
    Core API logic shared between sync and async clients.
    It's designed specifically for an OpenAI-compatible "chat/completions" endpoint.
    Which is a Tightly Coupled design, A better approach would be, 
    Decouple the generic API logic from the specific "chat" logic. 
    `APIExecutor` could be refactored into a `BaseAPIExecutor`, with a `ChatAPIExecutor` 
    subclass that manages message history and chat-specific data formatting.
    """

    def __init__(self, api_key: str, base_url: str, model: str, 
                 endpoint: str = "chat/completions",
                 temperature: float = 0.7,
                 max_completion_tokens: int = 100,
                 timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.endpoint = endpoint.rstrip('/').lstrip('/')
        self.timeout = timeout

        # LLM-specific parameters
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

    def prepare_request_data(self, prompt: str|List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Prepare the request data for the API call."""
        if isinstance(prompt, str):
            # Add user message to history
            self.add_message("user", prompt)
        elif isinstance(prompt, list):
            validate_messages_format(prompt)
            self.messages = prompt
        else:
            raise TypeError(f"Prompt must be either a string or a list of message dictionaries, got {type(prompt)!r}")


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

    def __init__(self, api_key: str, base_url: str, model: str,
                 endpoint: str = "chat/completions",
                 temperature: float = 0.7,
                 max_completion_tokens: int = 100,
                 timeout: float = 30.0,
                 http_client: Optional[SyncHTTPClient] = None,
                 middleware: Optional[List[BaseMiddleware]] = None,
                 connection_pool: Optional[ConnectionPool] = None,
                 retry_config: Optional[RetryConfig] = None):

        self._executor = APIExecutor(
            api_key, base_url, model, endpoint, temperature, max_completion_tokens, timeout
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

    def post(self, url: str, headers: Optional[Dict[str, str]] = None,
             body: Optional[bytes] = None, timeout: Optional[float] = None) -> HTTPResponse:
        """Send a synchronous POST request."""
        return self._http_client.request('POST', url, headers=headers, body=body, timeout=timeout)

    def send(self, messages_for_request: list[dict]) -> list[dict]:
        data = self._executor.prepare_request_data(messages_for_request, stream=False)
        url, headers, body = self._executor.get_request_config(data)
        response = self.post(url, headers=headers, body=body, timeout=self._executor.timeout)
        assistant_message = self._executor.process_non_streaming_response(response)
        return assistant_message

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

    def __init__(self, api_key: str, base_url: str, model: str,
                endpoint: str = "chat/completions",
                temperature: float = 0.7,
                max_completion_tokens: int = 100,
                timeout: float = 30.0,
                http_client: Optional[AsyncHTTPClient] = None,
                middleware: Optional[List[BaseMiddleware]] = None,
                connection_pool: Optional[ConnectionPool] = None,
                retry_config: Optional[RetryConfig] = None):

        self._executor = APIExecutor(
            api_key, base_url, model, endpoint, temperature, max_completion_tokens, timeout
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

    async def post(self, url: str, headers: Optional[Dict[str, str]] = None,
             body: Optional[bytes] = None, timeout: Optional[float] = None) -> HTTPResponse:
        """Send a asynchronous POST request."""
        return await self._http_client.request('POST', url, headers=headers, body=body, timeout=timeout)

    async def send(self, messages_for_request: list[dict]) -> list[dict]:
        data = self._executor.prepare_request_data(messages_for_request, stream=False)
        url, headers, body = self._executor.get_request_config(data)
        response = await self.post(url, headers=headers, body=body, timeout=self._executor.timeout)
        assistant_message = self._executor.process_non_streaming_response(response)
        return assistant_message

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
