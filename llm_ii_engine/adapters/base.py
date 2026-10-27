"""Asynchronous inference engine adapter protocol and abstract base class.

Defines the pluggable contract for all inference runtimes (mock, Ollama, vLLM, etc.)
independent of web frameworks, storage mechanisms, or UI layers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from llm_ii_engine.core.enums import EngineType, ExecutionMode
from llm_ii_engine.core.schemas import (
    EngineCapabilities,
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    TokenStreamEvent,
)


@runtime_checkable
class InferenceEngineAdapter(Protocol):
    """Structural protocol for asynchronous inference engine adapters.

    All engine adapters must conform to this interface to allow interchangeable
    use by benchmark runners, diagnostic modules, and CLI tools.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for the adapter instance."""
        ...

    @property
    def engine_type(self) -> EngineType:
        """Engine type enum identifying the runtime category."""
        ...

    @property
    def execution_mode(self) -> ExecutionMode:
        """Distinguishes real hardware execution from simulated mock execution."""
        ...

    def get_capabilities(self) -> EngineCapabilities:
        """Return static and dynamic capabilities supported by this adapter."""
        ...

    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool:
        """Initialize the adapter, load model weights, or establish connection.

        Args:
            model_spec: Target model architecture and parameters.
            engine_config: Engine runtime configuration parameters.

        Returns:
            True if initialization succeeded and the engine is ready for inference.
        """
        ...

    async def health_check(self) -> bool:
        """Verify whether the engine is responsive and able to service requests."""
        ...

    def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenStreamEvent]:
        """Asynchronously stream token events for a generation workload.

        Args:
            request: Validated generation request containing model, config, and workload.

        Yields:
            TokenStreamEvent instances representing the request lifecycle:
            START, intermediate TOKEN events, and a terminal COMPLETION or ERROR event.

        Raises:
            EngineExecutionError: If an unrecoverable runtime error occurs during generation.
            StreamCancelledError: If generation is cancelled by the caller.
        """
        ...

    async def get_memory_usage_mb(self) -> float | None:
        """Query current memory usage in megabytes (VRAM for GPU, RSS for CPU).

        Returns:
            Current memory consumption in MB, or None if the engine runtime cannot
            expose memory metrics.
        """
        ...

    async def teardown(self) -> None:
        """Release allocated resources, terminate worker tasks, or close network connections."""
        ...


class BaseInferenceAdapter(ABC):
    """Abstract base class offering shared lifecycle logic for engine adapters."""

    def __init__(self, name: str, engine_type: EngineType, execution_mode: ExecutionMode) -> None:
        self._name = name
        self._engine_type = engine_type
        self._execution_mode = execution_mode
        self._is_initialized = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine_type(self) -> EngineType:
        return self._engine_type

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """Return capabilities supported by this adapter."""

    @abstractmethod
    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool:
        """Initialize adapter."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check adapter health."""

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenStreamEvent]:
        """Stream generated tokens."""
        # Async generator stub
        if False:
            yield

    async def get_memory_usage_mb(self) -> float | None:
        """Default fallback: returns None if memory introspection is unsupported."""
        return None

    async def teardown(self) -> None:
        """Default teardown: mark uninitialized."""
        self._is_initialized = False
