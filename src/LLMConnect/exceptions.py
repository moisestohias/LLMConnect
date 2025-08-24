from typing import Dict, Optional

# Exceptions
class APIError(Exception):
    """Base exception for API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None,
                 headers: Optional[Dict[str, str]] = None, body: Optional[str] = None,
                 request_id: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body
        self.request_id = request_id
        self.retry_after = self._parse_retry_after()

    def _parse_retry_after(self) -> Optional[float]:
        """Parse Retry-After header."""
        retry_after = self.headers.get('retry-after') or self.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                return None
        return None

class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass

class TimeoutError(APIError):
    """Raised when request times out."""
    pass

class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass

class ConnectionError(APIError):
    """Raised when connection fails."""
    pass

class RetryableError(APIError):
    """Base class for errors that should be retried."""
    pass