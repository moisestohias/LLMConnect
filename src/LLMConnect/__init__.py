"""LLMConnect - A Python HTTP client for LLM providers."""

# Import key classes for easier access
from .top import SyncAPIClient, AsyncAPIClient
from .api_client_factory import (
    create_cerebras_sync_client,
    create_cerebras_async_client,
    create_groq_sync_client,
    create_groq_async_client,
    create_openrouter_sync_client,
    create_openrouter_async_client,
    APIClientFactory,
    Provider
)

__version__ = "0.1.0"
__author__ = "Moises-Tohias"