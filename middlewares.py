import logging
import threading
# Configure logging
logger = logging.getLogger(__name__)

from typing import Dict, Optional, Any

from .models import *

# Middleware System
class BaseMiddleware:
    """Base class for HTTP middleware."""

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        """Process the request before it's sent."""
        return request

    async def process_response(self, response: HTTPResponse) -> HTTPResponse:
        """Process the response after it's received."""
        return response

    async def process_error(self, error: Exception, request: HTTPRequest) -> Exception:
        """Process an error that occurred during the request."""
        return error

class LoggingMiddleware(BaseMiddleware):
    """Middleware for logging requests and responses."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        self.logger.debug(f"Request: {request.method} {request.url}")
        return request

    async def process_response(self, response: HTTPResponse) -> HTTPResponse:
        self.logger.debug(f"Response: {response.status_code} ({response.elapsed:.3f}s)")
        return response

    async def process_error(self, error: Exception, request: HTTPRequest) -> Exception:
        self.logger.error(f"Request failed: {request.method} {request.url} - {error}")
        return error

class AuthenticationMiddleware(BaseMiddleware):
    """Middleware for adding authentication headers."""

    def __init__(self, token: str, auth_type: str = "Bearer"):
        self.token = token
        self.auth_type = auth_type

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        if 'Authorization' not in request.headers:
            request.headers['Authorization'] = f"{self.auth_type} {self.token}"
        return request

class UserAgentMiddleware(BaseMiddleware):
    """Middleware for adding User-Agent header."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        if 'User-Agent' not in request.headers:
            request.headers['User-Agent'] = self.user_agent
        return request

class MetricsMiddleware(BaseMiddleware):
    """Middleware for collecting request metrics."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self._lock = threading.Lock()

    async def process_response(self, response: HTTPResponse) -> HTTPResponse:
        with self._lock:
            self.request_count += 1
            self.total_response_time += response.elapsed
        return response

    async def process_error(self, error: Exception, request: HTTPRequest) -> Exception:
        with self._lock:
            self.error_count += 1
        return error

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'average_response_time': (
                    self.total_response_time / self.request_count
                    if self.request_count > 0 else 0.0
                ),
                'error_rate': (
                    self.error_count / (self.request_count + self.error_count)
                    if (self.request_count + self.error_count) > 0 else 0.0
                )
            }

