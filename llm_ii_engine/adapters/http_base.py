"""Base class and common infrastructure for HTTP streaming inference engine adapters.

Provides connection management, timeout configuration, dependency injection of HTTP
transports/clients, and lifecycle management for production-conscious adapters.
"""

import httpx

from llm_ii_engine.adapters.base import BaseInferenceAdapter
from llm_ii_engine.core.enums import EngineType, ExecutionMode
from llm_ii_engine.core.exceptions import (
    EngineConnectionError,
    EngineExecutionError,
    EngineTimeoutError,
    ModelNotFoundError,
)


class BaseHttpEngineAdapter(BaseInferenceAdapter):
    """Abstract base adapter for HTTP streaming inference engines (Ollama, llama.cpp, etc.).

    Supports dependency injection of custom httpx.AsyncClient or httpx.AsyncBaseTransport
    to enable offline integration testing without live external servers.
    """

    def __init__(
        self,
        name: str,
        engine_type: EngineType,
        base_url: str,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the HTTP engine adapter.

        Args:
            name: Human-readable name for this adapter instance.
            engine_type: EngineType classification.
            base_url: Target server base URL (e.g. 'http://localhost:11434').
            timeout_seconds: Overall request and stream read timeout in seconds.
            connect_timeout_seconds: Initial TCP connection timeout in seconds.
            client: Optional injected httpx.AsyncClient (for offline tests/mocking).
            transport: Optional injected httpx.AsyncBaseTransport (for offline tests/mocking).
        """
        super().__init__(name=name, engine_type=engine_type, execution_mode=ExecutionMode.REAL)
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._connect_timeout_seconds = connect_timeout_seconds
        self._transport = transport
        self._client = client
        self._owns_client = client is None

    @property
    def base_url(self) -> str:
        """Return the target server base URL."""
        return self._base_url

    def get_client(self) -> httpx.AsyncClient:
        """Retrieve or lazily instantiate the HTTP client."""
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                self._timeout_seconds,
                connect=self._connect_timeout_seconds,
            )
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
                transport=self._transport,
            )
            self._owns_client = True
        return self._client

    def _map_http_error(self, exc: Exception, endpoint: str) -> None:
        """Map common httpx networking exceptions into domain-specific exceptions."""
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            raise EngineConnectionError(
                f"Could not connect to {self.name} server at {self._base_url}{endpoint}. "
                f"Is the server running and accessible? Details: {exc}"
            ) from exc
        if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            raise EngineTimeoutError(
                f"Timeout communicating with {self.name} server at {self._base_url}{endpoint} "
                f"after {self._timeout_seconds}s. Details: {exc}"
            ) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            text = exc.response.text
            if status == 404:
                raise ModelNotFoundError(
                    f"{self.name} reported resource or model not found (404) at {endpoint}: {text}"
                ) from exc
            raise EngineExecutionError(
                f"{self.name} server returned HTTP {status} error at {endpoint}: {text}"
            ) from exc

    async def teardown(self) -> None:
        """Release HTTP connections and mark adapter uninitialized."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        await super().teardown()
