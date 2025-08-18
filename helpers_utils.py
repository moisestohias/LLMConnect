from top import *

import os
api_key: str = os.getenv("CEREBRAS_API_KEY")

# base Factory Functions
# def create_sync_client(connection_pool: Optional[ConnectionPool] = None,
#                       retry_config: Optional[RetryConfig] = None,
#                       middleware: Optional[List[BaseMiddleware]] = None) -> SyncHTTPClient:
#     """Create a configured synchronous HTTP client."""
#     if middleware is None:
#         middleware = [
#             UserAgentMiddleware(user_agent),
#             LoggingMiddleware(),
#         ]
#         if api_key:
#             middleware.insert(-1, AuthenticationMiddleware(api_key))

#     return SyncHTTPClient(
#         connection_pool=connection_pool,
#         retry_config=retry_config,
#         middleware=middleware
#     )
# def create_async_client(connection_pool: Optional[ConnectionPool] = None,
#                        retry_config: Optional[RetryConfig] = None,
#                        middleware: Optional[List[BaseMiddleware]] = None) -> AsyncHTTPClient:
#     """Create a configured asynchronous HTTP client."""
#     if middleware is None:
#         middleware = [
#             UserAgentMiddleware(user_agent),
#             LoggingMiddleware(),
#         ]
#         if api_key:
#             middleware.insert(-1, AuthenticationMiddleware(api_key))

#     return AsyncHTTPClient(
#         connection_pool=connection_pool,
#         retry_config=retry_config,
#         middleware=middleware
#     )

# top Factory Functions
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

