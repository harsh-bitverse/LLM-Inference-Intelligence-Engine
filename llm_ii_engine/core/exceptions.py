"""Domain-specific exceptions for LLM Inference Intelligence Engine."""


class IIEError(Exception):
    """Base exception for all errors in the LLM Inference Intelligence Engine."""


class ConfigurationError(IIEError):
    """Raised when an invalid configuration or constraint specification is supplied."""


class EngineExecutionError(IIEError):
    """Raised when an inference engine encounters an unrecoverable runtime error."""


class EngineTimeoutError(EngineExecutionError):
    """Raised when an inference generation request exceeds its allotted deadline."""


class EngineConnectionError(EngineExecutionError):
    """Raised when connection to an external or local engine adapter fails."""


class ModelNotFoundError(EngineExecutionError):
    """Raised when the requested model is not found or loaded on the engine server."""


class StreamCancelledError(IIEError):
    """Raised or propagated when an active token stream is cancelled by the client."""
