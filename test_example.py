import os
api_key: str = os.getenv("CEREBRAS_API_KEY")

from helpers_utils import create_cerebras_sync_client, create_cerebras_async_client, create_groq_sync_client, asyncio

# Example usage functions
def sync_example():
    """Example of using the synchronous API client."""
    print("=== Synchronous API Client Example ===")
    
    # with create_cerebras_sync_client() as client:
    with create_groq_sync_client() as client:
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
    sync_example()

    # Test async client
    # asyncio.run(async_example())

if __name__ == "__main__":
    main()
