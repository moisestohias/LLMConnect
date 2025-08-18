from dataclasses import dataclass, field
from typing import Dict, Optional
from urllib.parse import urlparse

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