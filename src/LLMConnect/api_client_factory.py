""" API Client Factory System with Provider Configuration """

import os, json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type, Union, Any, Callable

from .top import SyncAPIClient, AsyncAPIClient # import BaseSyncHTTPAPIClient, BaseAsyncHTTPAPIClient


class Provider(Enum):
    """Enum for supported API providers."""
    CEREBRAS = "cerebras"
    GROQ = "groq"
    OPENROUTER = "openrouter"

@dataclass
class ProviderConfig:
    """Base configuration for an API provider."""
    name: str
    base_url: str
    endpoint: str
    api_key_env_var: str
    available_models: List[str]
    default_model: str
    default_temperature: float = 0.7
    default_max_tokens: int = 100
    default_timeout: float = 30.0
    
    def __post_init__(self):
        """Validate that default model is in available models."""
        if self.default_model not in self.available_models:
            raise ValueError(f"Default model '{self.default_model}' not in available models for {self.name}")
    
    def validate_model(self, model: str) -> bool:
        """Check if the given model is supported by this provider."""
        return model in self.available_models
    
    def get_api_key(self, provided_key: Optional[str] = None) -> str:
        """Get API key from provided key or environment variable."""
        if provided_key:
            return provided_key
        
        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key not found. Please provide it or set {self.api_key_env_var} environment variable.")
        return api_key


def load_provider_configs() -> Dict[Provider, ProviderConfig]:
    """Load provider configurations from JSON file."""
    config_file_path = os.path.join(os.path.dirname(__file__), 'provider_configs.json')
    
    try:
        with open(config_file_path, 'r') as f: raw_configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Provider configuration file not found at {config_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in provider configuration file: {e}")
    
    provider_configs = {}
    for provider_key, config_data in raw_configs.items():
        try:
            provider_type = Provider(provider_key)
            provider_configs[provider_type] = ProviderConfig(**config_data)
        except ValueError:
            # Skip unknown providers
            continue
    
    return provider_configs



class APIClientFactory:
    """Factory class for creating API clients with provider-specific configurations."""
    
    @staticmethod
    def _validate_model_for_provider(provider: Provider, model: str) -> None:
        """Validate that the model is supported by the provider."""
        config = PROVIDER_CONFIGS[provider]
        if not config.validate_model(model):
            available_models_str = ", ".join(config.available_models)
            raise ValueError(
                f"Model '{model}' is not supported by {config.name}. "
                f"Available models: {available_models_str}"
            )
    
    @staticmethod
    def create_sync_client(
        provider: Provider,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> SyncAPIClient:
        """Create a synchronous API client for the specified provider."""
        config = PROVIDER_CONFIGS[provider]
        
        # Use default model if not provided
        model = model or config.default_model
        
        # Validate model
        APIClientFactory._validate_model_for_provider(provider, model)
        
        return SyncAPIClient(
            api_key=config.get_api_key(api_key),
            base_url=config.base_url,
            model=model,
            endpoint=config.endpoint,
            temperature=temperature or config.default_temperature,
            max_completion_tokens=max_completion_tokens or config.default_max_tokens,
            timeout=timeout or config.default_timeout,
            **kwargs
        )
    
    @staticmethod
    def create_async_client(
        provider: Provider,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncAPIClient:
        """Create an asynchronous API client for the specified provider."""
        config = PROVIDER_CONFIGS[provider]
        
        # Use default model if not provided
        model = model or config.default_model
        
        # Validate model
        APIClientFactory._validate_model_for_provider(provider, model)
        
        return AsyncAPIClient(
            api_key=config.get_api_key(api_key),
            base_url=config.base_url,
            model=model,
            endpoint=config.endpoint,
            temperature=temperature or config.default_temperature,
            max_completion_tokens=max_completion_tokens or config.default_max_tokens,
            timeout=timeout or config.default_timeout,
            **kwargs
        )
    
    @staticmethod
    def get_available_models(provider: Provider) -> List[str]:
        """Get list of available models for a provider."""
        return PROVIDER_CONFIGS[provider].available_models.copy()
    
    @staticmethod
    def get_provider_info(provider: Provider) -> Dict[str, Any]:
        """Get comprehensive information about a provider."""
        config = PROVIDER_CONFIGS[provider]
        return {
            "name": config.name,
            "base_url": config.base_url,
            "endpoint": config.endpoint,
            "api_key_env_var": config.api_key_env_var,
            "available_models": config.available_models.copy(),
            "default_model": config.default_model,
            "default_temperature": config.default_temperature,
            "default_max_tokens": config.default_max_tokens,
            "default_timeout": config.default_timeout
        }


# DRY approach: Generate convenience functions dynamically (inspired by user's approach)
def _make_sync_client_func(provider: Provider) -> Callable:
    """Create a sync client factory function for a specific provider."""
    def create_sync_client(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> SyncAPIClient:
        return APIClientFactory.create_sync_client(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs
        )
    
    provider_name = provider.value
    create_sync_client.__name__ = f"create_{provider_name}_sync_client"
    create_sync_client.__doc__ = f"Create a synchronous {provider_name.capitalize()} API client."
    create_sync_client.__annotations__ = {
        'model': Optional[str],
        'api_key': Optional[str],
        'return': SyncAPIClient
    }
    return create_sync_client


def _make_async_client_func(provider: Provider) -> Callable:
    """Create an async client factory function for a specific provider."""
    def create_async_client(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> AsyncAPIClient:
        return APIClientFactory.create_async_client(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs
        )
    
    provider_name = provider.value
    create_async_client.__name__ = f"create_{provider_name}_async_client"
    create_async_client.__doc__ = f"Create an asynchronous {provider_name.capitalize()} API client."
    create_async_client.__annotations__ = {
        'model': Optional[str],
        'api_key': Optional[str],
        'return': AsyncAPIClient
    }
    return create_async_client



# Provider configurations
PROVIDER_CONFIGS = load_provider_configs()

# Dynamically generate and assign convenience functions to the global namespace
for provider in PROVIDER_CONFIGS.keys():
    sync_func = _make_sync_client_func(provider)
    async_func = _make_async_client_func(provider)
    
    globals()[sync_func.__name__] = sync_func
    globals()[async_func.__name__] = async_func

## !WARNING: Be carefull here, if you want to define __all__ you must include all stuff above that must be imported in order to be able to use the functionality of this module.

# __all__ = [
#     "create_cerebras_sync_client",
#     "create_cerebras_async_client",
#     "create_groq_sync_client",
#     "create_groq_async_client",
#     "create_openrouter_sync_client",
#     "create_openrouter_async_client",
# ]

# # --- New Factory Functions for Base Clients ---

# def create_base_sync_client(
#     provider: Provider,
#     api_key: Optional[str] = None,
#     **kwargs
# ) -> BaseSyncHTTPAPIClient:
#     """
#     Create a base synchronous HTTP API client for the specified provider.
#     This client provides low-level HTTP methods without chat-specific features.
#     """
#     config = PROVIDER_CONFIGS[provider]
#     return BaseSyncHTTPAPIClient(
#         api_key=config.get_api_key(api_key),
#         **kwargs
#     )


# def create_base_async_client(
#     provider: Provider,
#     api_key: Optional[str] = None,
#     **kwargs
# ) -> BaseAsyncHTTPAPIClient:
#     """
#     Create a base asynchronous HTTP API client for the specified provider.
#     This client provides low-level HTTP methods without chat-specific features.
#     """
#     config = PROVIDER_CONFIGS[provider]
#     return BaseAsyncHTTPAPIClient(
#         api_key=config.get_api_key(api_key),
#         **kwargs
#     )

def main():
  """ tests """
  client = create_groq_sync_client()

  user_prompt_1 = "say hello and nothing else"
  user_prompt_2 = "now say it Spanish"

  # ## Auto-Conversation Management Using Chat
  # resp = client.chat(user_prompt_1)
  # print(resp)
  # resp = client.chat(user_prompt_2)
  # print(resp)
  # print(client.messages)

  # ## Manual-Conversation Management Using Send
  # messages = []
  # def append_user_prompt(prompt): messages.append({'role': 'user', 'content': prompt})
  # def append_assistant_prompt(prompt): messages.append({'role': 'assistant', 'content': prompt})
  # # 
  # append_user_prompt(user_prompt_1)
  # resp = client.send(messages)
  # append_assistant_prompt(resp)
  # print(resp)

  # # 
  # append_user_prompt(user_prompt_2)
  # resp = client.send(messages)
  # append_assistant_prompt(resp)
  # print(resp)

if __name__ == '__main__':
  main()