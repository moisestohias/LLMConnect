
def validate_messages_format(messages):
  # Validate the conversations list format
  if not messages:
      raise ValueError("Conversations list cannot be empty")
  
  for i, message in enumerate(messages):
      if not isinstance(message, dict):
          raise ValueError(f"Message at index {i} must be a dictionary")
      
      if 'role' not in message or 'content' not in message:
          raise ValueError(f"Message at index {i} must contain 'role' and 'content' keys")
      
      if not isinstance(message['role'], str) or not isinstance(message['content'], str):
          raise ValueError(f"Message at index {i} 'role' and 'content' must be strings")
      
      if message['role'] not in ['user', 'assistant', 'system']:
          raise ValueError(f"Message at index {i} has invalid role '{message['role']}'. Must be 'user', 'assistant', or 'system'")
