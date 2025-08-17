import asyncio, time, ssl, socket, http.client, threading, random
from contextlib import contextmanager

import os
from typing import Dict, List, Optional, Any, AsyncIterator, Tuple

api_key: Optional[str] = os.getenv("CEREBRAS_API_KEY")
user_agent: str = "APIClient/1.0.0"
from dataclasses import dataclass, field
from urllib.parse import urlparse
import logging

# Configure logging
logger = logging.getLogger(__name__)

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

# Request/Response Models
@dataclass
class HTTPRequest:
    """Represents an HTTP request."""
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    timeout: float = 30.0

    @property
    def parsed_url(self):
        return urlparse(self.url)

@dataclass
class HTTPResponse:
    """Represents an HTTP response."""
    status_code: int
    headers: Dict[str, str]
    body: bytes
    request: HTTPRequest
    elapsed: float = 0.0

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

# Connection Management
class ConnectionPool:
    """Thread-safe HTTP connection pool."""

    def __init__(self, max_connections_per_host: int = 10):
        self.max_connections_per_host = max_connections_per_host
        self._connections: Dict[str, List[http.client.HTTPConnection]] = {}
        self._lock = threading.Lock()

    def _get_pool_key(self, parsed_url) -> str:
        """Get the pool key for a URL."""
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    def _create_connection(self, parsed_url) -> http.client.HTTPConnection:
        """Create a new connection for the given URL."""
        if parsed_url.scheme == 'https':
            conn = http.client.HTTPSConnection(
                parsed_url.hostname,
                parsed_url.port,
                context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(
                parsed_url.hostname,
                parsed_url.port
            )
        return conn

    @contextmanager
    def get_connection(self, parsed_url):
        """Get a connection from the pool."""
        pool_key = self._get_pool_key(parsed_url)
        connection = None

        with self._lock:
            # Try to get an existing connection
            if pool_key in self._connections and self._connections[pool_key]:
                connection = self._connections[pool_key].pop(0)

        # Create new connection if needed
        if connection is None:
            connection = self._create_connection(parsed_url)

        try:
            yield connection
        finally:
            # Return connection to pool
            with self._lock:
                if pool_key not in self._connections:
                    self._connections[pool_key] = []

                # Only return to pool if not full and connection is still good
                if len(self._connections[pool_key]) < self.max_connections_per_host:
                    try:
                        # Check if connection is still good
                        if hasattr(connection, 'sock') and connection.sock:
                            # Try to get connection state
                            connection.sock.settimeout(0.1)
                            try:
                                data = connection.sock.recv(1, socket.MSG_PEEK)
                                # If we get here without exception, connection might be dead
                                if not data:
                                    connection.close()
                                else:
                                    self._connections[pool_key].append(connection)
                            except socket.error:
                                # Connection is still alive (would block)
                                self._connections[pool_key].append(connection)
                            finally:
                                connection.sock.settimeout(None)
                        else:
                            # Connection not established yet, can reuse
                            self._connections[pool_key].append(connection)
                    except Exception:
                        # If any error, just close the connection
                        connection.close()
                else:
                    connection.close()

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for connections in self._connections.values():
                for conn in connections:
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._connections.clear()

# Retry Logic
class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 backoff_factor: float = 2.0, jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if a request should be retried."""
        if attempt >= self.max_retries:
            return False

        # Retry on specific error types
        if isinstance(error, (TimeoutError, ConnectionError, RetryableError)):
            return True

        # Retry on rate limit errors
        if isinstance(error, RateLimitError):
            return True

        # Retry on specific HTTP status codes
        if isinstance(error, APIError) and error.status_code:
            return error.status_code in [408, 429] or error.status_code >= 500

        return False

    def get_delay(self, attempt: int, error: Optional[Exception] = None) -> float:
        """Calculate the delay before retrying."""
        # Use Retry-After header if available
        if isinstance(error, APIError) and error.retry_after:
            return error.retry_after

        # Calculate exponential backoff delay
        delay = self.base_delay * (self.backoff_factor ** attempt)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

# Core Request Execution Logic
class RequestExecutor:
    """Core request execution logic shared between sync and async clients."""

    def __init__(self, connection_pool: Optional[ConnectionPool] = None,
                 retry_config: Optional[RetryConfig] = None,
                 middleware: Optional[List[BaseMiddleware]] = None):
        self.connection_pool = connection_pool or ConnectionPool()
        self.retry_config = retry_config or RetryConfig()
        self.middleware = middleware or []
        self._closed = False

    async def execute_request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute an HTTP request with retry logic and middleware."""
        if self._closed:
            raise RuntimeError("Client is closed")

        # Process request through middleware
        for middleware in self.middleware:
            request = await middleware.process_request(request)

        last_error = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                response = await self._execute_single_request(request)

                # Process response through middleware
                for middleware in reversed(self.middleware):
                    response = await middleware.process_response(response)

                return response

            except Exception as error:
                # Process error through middleware
                for middleware in self.middleware:
                    error = await middleware.process_error(error, request)

                last_error = error

                if not self.retry_config.should_retry(attempt, error):
                    break

                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt, error)
                    logger.debug(f"Retrying request in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("No attempts were made")

    async def execute_streaming_request(self, request: HTTPRequest, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        """Execute a streaming HTTP request with retry logic."""
        if self._closed:
            raise RuntimeError("Client is closed")

        # Process request through middleware
        for middleware in self.middleware:
            request = await middleware.process_request(request)

        last_error = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                async for chunk in self._execute_single_streaming_request(request, chunk_size):
                    yield chunk
                return  # Successfully completed streaming

            except Exception as error:
                # Process error through middleware
                for middleware in self.middleware:
                    error = await middleware.process_error(error, request)

                last_error = error

                if not self.retry_config.should_retry(attempt, error):
                    break

                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt, error)
                    logger.debug(f"Retrying streaming request in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("No streaming attempts were made")

    async def _execute_single_request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute a single HTTP request."""
        start_time = time.time()
        loop = asyncio.get_event_loop()

        try:
            status_code, headers, body = await loop.run_in_executor(
                None, self._sync_request, request
            )

            elapsed = time.time() - start_time

            response = HTTPResponse(
                status_code=status_code,
                headers=headers,
                body=body,
                request=request,
                elapsed=elapsed
            )

            # Check for HTTP errors
            if status_code >= 400:
                error_body = body.decode('utf-8', errors='ignore')

                if status_code == 401:
                    raise AuthenticationError(
                        f"Authentication failed: {status_code}",
                        status_code=status_code,
                        headers=headers,
                        body=error_body
                    )
                elif status_code == 429:
                    raise RateLimitError(
                        f"Rate limit exceeded: {status_code}",
                        status_code=status_code,
                        headers=headers,
                        body=error_body
                    )
                elif status_code >= 500:
                    raise RetryableError(
                        f"Server error: {status_code}",
                        status_code=status_code,
                        headers=headers,
                        body=error_body
                    )
                else:
                    raise APIError(
                        f"HTTP error: {status_code}",
                        status_code=status_code,
                        headers=headers,
                        body=error_body
                    )
            return response

        except (socket.timeout, socket.error) as e:
            raise ConnectionError(f"Connection failed: {str(e)}")
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError(f"Request failed: {str(e)}")
            raise

    async def _execute_single_streaming_request(self, request: HTTPRequest, chunk_size: int) -> AsyncIterator[bytes]:
        """Execute a single streaming HTTP request."""
        loop = asyncio.get_event_loop()
        parsed_url = request.parsed_url
        path = parsed_url.path or '/'
        if parsed_url.query:
            path += '?' + parsed_url.query

        try:
            with self.connection_pool.get_connection(parsed_url) as conn:
                conn.timeout = request.timeout
                conn.connect()

                # Send request with headers and body
                conn.putrequest(request.method, path)

                # Send headers
                for header_name, header_value in request.headers.items():
                    conn.putheader(header_name, header_value)

                # End headers and send body if present
                if request.body:
                    conn.putheader('Content-Length', str(len(request.body)))
                    conn.endheaders()
                    conn.send(request.body)
                else:
                    conn.endheaders()

                # Get response
                response = conn.getresponse()

                # Check status code
                if response.status >= 400:
                    error_body = response.read().decode('utf-8', errors='ignore')
                    headers = dict(response.headers)

                    if response.status == 401:
                        raise AuthenticationError(
                            f"Authentication failed: {response.status}",
                            status_code=response.status,
                            headers=headers,
                            body=error_body
                        )
                    elif response.status == 429:
                        raise RateLimitError(
                            f"Rate limit exceeded: {response.status}",
                            status_code=response.status,
                            headers=headers,
                            body=error_body
                        )
                    else:
                        raise APIError(
                            f"HTTP error: {response.status}",
                            status_code=response.status,
                            headers=headers,
                            body=error_body
                        )

                # Check if this is an SSE stream
                content_type = response.headers.get('content-type', '')
                is_sse = 'text/event-stream' in content_type
                
                # For SSE, we need to read line by line to properly handle events
                if is_sse:
                    buffer = b''
                    while True:
                        # Read a smaller chunk for SSE to avoid buffering issues
                        chunk = await loop.run_in_executor(None, response.read, 1024)
                        if not chunk:
                            # Yield any remaining buffer
                            if buffer:
                                yield buffer
                            break
                        
                        # Add to buffer
                        buffer += chunk
                        
                        # Process complete lines
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            # Yield complete line with newline
                            yield line + b'\n'
                        
                        # Check if we have a complete SSE end marker in buffer
                        if b'data: [DONE]' in buffer:
                            yield buffer
                            break
                else:
                    # For non-SSE streams, use the original chunking approach
                    while True:
                        chunk = await loop.run_in_executor(None, response.read, chunk_size)
                        if not chunk:
                            break
                        yield chunk

        except socket.timeout:
            raise TimeoutError(f"Streaming request timed out after {request.timeout} seconds")
        except socket.error as e:
            raise ConnectionError(f"Streaming connection error: {str(e)}")
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError(f"Streaming request failed: {str(e)}")
            raise

    def _sync_request(self, request: HTTPRequest) -> Tuple[int, Dict[str, str], bytes]:
        """Execute synchronous HTTP request using connection pool."""
        parsed_url = request.parsed_url
        path = parsed_url.path or '/'
        if parsed_url.query:
            path += '?' + parsed_url.query

        try:
            with self.connection_pool.get_connection(parsed_url) as conn:
                # Set timeout
                conn.timeout = request.timeout

                # Connect explicitly to ensure connection is established
                conn.connect()

                # Send request with headers and body
                conn.putrequest(request.method, path)

                # Send headers
                for header_name, header_value in request.headers.items():
                    conn.putheader(header_name, header_value)

                # End headers and send body if present
                if request.body:
                    conn.putheader('Content-Length', str(len(request.body)))
                    conn.endheaders()
                    conn.send(request.body)
                else:
                    conn.endheaders()

                # Get response
                response = conn.getresponse()

                # Read response data
                body = response.read()

                # Convert headers to dict
                headers = dict(response.headers)

                return response.status, headers, body

        except socket.timeout:
            raise TimeoutError(f"Request timed out after {request.timeout} seconds")
        except socket.error as e:
            raise ConnectionError(f"Connection error: {str(e)}")
        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")

    def close(self):
        """Close the executor and all connections."""
        self._closed = True
        self.connection_pool.close_all()

# Synchronous Client
class SyncHTTPClient:
    """Synchronous HTTP client with connection pooling, retry logic, and middleware."""

    def __init__(self, connection_pool: Optional[ConnectionPool] = None,
                 retry_config: Optional[RetryConfig] = None,
                 middleware: Optional[List[BaseMiddleware]] = None):
        self._executor = RequestExecutor(connection_pool, retry_config, middleware)
        pass

    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
                body: Optional[bytes] = None, timeout: float = 30.0) -> HTTPResponse:
        """Make a synchronous HTTP request with retry logic."""
        request = HTTPRequest(
            method=method,
            url=url,
            headers=headers or {},
            body=body,
            timeout=timeout
        )

        return self._run_async(self._executor.execute_request(request))

    def stream_request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
                      body: Optional[bytes] = None, timeout: float = 30.0,
                      chunk_size: int = 8192):
        """Make a synchronous streaming HTTP request."""
        request = HTTPRequest(
            method=method,
            url=url,
            headers=headers or {},
            body=body,
            timeout=timeout
        )

        async def async_generator():
            async for chunk in self._executor.execute_streaming_request(request, chunk_size):
                yield chunk

        return self._run_async_generator(async_generator())

    def _run_async(self, coro):
        """Run an async coroutine in sync context."""
        try:
            asyncio.get_running_loop()
            # If we're already in an event loop, we need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(coro)

    def _run_async_generator(self, async_gen):
        """Convert async generator to sync iterator."""
        try:
            asyncio.get_running_loop()
            # We're in an event loop - collect all items in a thread
            import queue
            import threading

            result_queue = queue.Queue()

            def collect_in_thread():
                try:
                    async def collect():
                        async for item in async_gen:
                            result_queue.put(('item', item))
                        result_queue.put(('done', None))
                    asyncio.run(collect())
                except Exception as e:
                    result_queue.put(('error', e))

            thread = threading.Thread(target=collect_in_thread)
            thread.start()

            while True:
                msg_type, value = result_queue.get()
                if msg_type == 'item':
                    yield value
                elif msg_type == 'done':
                    break
                elif msg_type == 'error':
                    raise value

            thread.join()

        except RuntimeError:
            # No event loop - can run directly
            items = []

            async def collect():
                async for item in async_gen:
                    items.append(item)

            asyncio.run(collect())

            for item in items:
                yield item

    def close(self):
        """Close the client and all connections."""
        self._executor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Asynchronous Client
class AsyncHTTPClient:
    """Asynchronous HTTP client with connection pooling, retry logic, and middleware."""

    def __init__(self, connection_pool: Optional[ConnectionPool] = None,
                 retry_config: Optional[RetryConfig] = None,
                 middleware: Optional[List[BaseMiddleware]] = None):
        self._executor = RequestExecutor(connection_pool, retry_config, middleware)

    async def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
                     body: Optional[bytes] = None, timeout: float = 30.0) -> HTTPResponse:
        """Make an asynchronous HTTP request with retry logic."""
        request = HTTPRequest(
            method=method,
            url=url,
            headers=headers or {},
            body=body,
            timeout=timeout
        )

        return await self._executor.execute_request(request)

    async def stream_request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
                            body: Optional[bytes] = None, timeout: float = 30.0,
                            chunk_size: int = 8192) -> AsyncIterator[bytes]:
        """Make an asynchronous streaming HTTP request."""
        request = HTTPRequest(
            method=method,
            url=url,
            headers=headers or {},
            body=body,
            timeout=timeout
        )

        async for chunk in self._executor.execute_streaming_request(request, chunk_size):
            yield chunk

    async def close(self):
        """Close the client and all connections."""
        self._executor.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Factory Functions
def create_sync_client(connection_pool: Optional[ConnectionPool] = None,
                      retry_config: Optional[RetryConfig] = None,
                      middleware: Optional[List[BaseMiddleware]] = None) -> SyncHTTPClient:
    """Create a configured synchronous HTTP client."""
    if middleware is None:
        middleware = [
            UserAgentMiddleware(user_agent),
            LoggingMiddleware(),
        ]
        if api_key:
            middleware.insert(-1, AuthenticationMiddleware(api_key))

    return SyncHTTPClient(
        connection_pool=connection_pool,
        retry_config=retry_config,
        middleware=middleware
    )

def create_async_client(connection_pool: Optional[ConnectionPool] = None,
                       retry_config: Optional[RetryConfig] = None,
                       middleware: Optional[List[BaseMiddleware]] = None) -> AsyncHTTPClient:
    """Create a configured asynchronous HTTP client."""
    if middleware is None:
        middleware = [
            UserAgentMiddleware(user_agent),
            LoggingMiddleware(),
        ]
        if api_key:
            middleware.insert(-1, AuthenticationMiddleware(api_key))

    return AsyncHTTPClient(
        connection_pool=connection_pool,
        retry_config=retry_config,
        middleware=middleware
    )

# Backward compatibility aliases
HTTPClient = AsyncHTTPClient
StreamingHTTPClient = AsyncHTTPClient  # Both use the same client now
create_http_client = create_async_client
