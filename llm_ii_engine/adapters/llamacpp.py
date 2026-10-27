"""llama.cpp server HTTP streaming inference engine adapter.

Interfaces with the llama.cpp HTTP server's /completion endpoint via Server-Sent
Events (SSE), measuring client-observed chunk arrivals using monotonic clocks,
and capturing server-side timings without confusing text chunks with discrete tokens.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_ii_engine.adapters.http_base import BaseHttpEngineAdapter
from llm_ii_engine.benchmark.workload import resolve_prompt_text
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    QuantizationType,
    StreamTerminationReason,
    TokenEventType,
)
from llm_ii_engine.core.exceptions import (
    EngineConnectionError,
    EngineExecutionError,
    EngineTimeoutError,
    StreamCancelledError,
)
from llm_ii_engine.core.schemas import (
    EngineCapabilities,
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    ServerReportedStats,
    TokenStreamEvent,
)


class LlamaCppAdapter(BaseHttpEngineAdapter):
    """Production-conscious HTTP streaming adapter for llama.cpp server."""

    DEFAULT_BASE_URL = "http://localhost:8080"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str = "LlamaCppAdapter",
    ) -> None:
        """Initialize llama.cpp adapter.

        Args:
            base_url: llama.cpp server base URL (defaults to http://localhost:8080).
            timeout_seconds: Overall request/stream timeout.
            connect_timeout_seconds: Connection timeout.
            client: Optional injected HTTP client.
            transport: Optional injected HTTP transport.
            name: Adapter identifier.
        """
        super().__init__(
            name=name,
            engine_type=EngineType.LLAMACPP,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            client=client,
            transport=transport,
        )
        self._model_spec: ModelSpec | None = None
        self._engine_config: EngineConfig | None = None
        self._server_props: dict[str, Any] = {}

    def get_capabilities(self) -> EngineCapabilities:
        """Return llama.cpp server capabilities."""
        return EngineCapabilities(
            engine_type=EngineType.LLAMACPP,
            execution_mode=ExecutionMode.REAL,
            supported_quantizations=[
                QuantizationType.NONE,
                QuantizationType.GGUF_Q4_K_M,
                QuantizationType.GGUF_Q8_0,
                QuantizationType.INT4,
                QuantizationType.INT8,
                QuantizationType.FP16,
            ],
            supports_streaming=True,
            supports_gpu_memory_query=False,
            supports_token_ids=True,  # Exposed when configured with token IDs in response
            supports_true_token_itl=False,  # Default /completion streams text chunks unless token IDs enabled
            supports_server_metrics=True,  # Server emits timings dictionary in terminal message
            max_context_window=self._server_props.get("default_generation_settings", {}).get(
                "n_ctx"
            ),
        )

    async def health_check(self) -> bool:
        """Verify llama.cpp server health via /health endpoint."""
        client = self.get_client()
        try:
            resp = await client.get("/health")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data.get("status") == "ok"
                return True
            return False
        except Exception:
            return False

    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool:
        """Initialize adapter and verify llama.cpp server status.

        Args:
            model_spec: Target model specification.
            engine_config: Engine configuration parameters.

        Raises:
            EngineConnectionError: If llama.cpp server is unreachable or loading a model.
        """
        client = self.get_client()

        # 1. Check health
        try:
            health_resp = await client.get("/health")
            if health_resp.status_code != 200:
                raise EngineConnectionError(
                    f"llama.cpp server at {self._base_url} returned HTTP {health_resp.status_code} on /health"
                )
            data = health_resp.json()
            if isinstance(data, dict) and data.get("status") == "loading model":
                raise EngineConnectionError(
                    f"llama.cpp server at {self._base_url} is still loading a model into memory"
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self._map_http_error(exc, "/health")
        except Exception as exc:
            if not isinstance(exc, EngineConnectionError):
                raise EngineConnectionError(
                    f"Failed to communicate with llama.cpp server at {self._base_url}: {exc}"
                ) from exc
            raise

        # 2. Query /props for server properties if supported
        try:
            props_resp = await client.get("/props")
            if props_resp.status_code == 200:
                self._server_props = props_resp.json()
        except Exception:
            self._server_props = {}

        self._model_spec = model_spec
        self._engine_config = engine_config
        self._is_initialized = True
        return True

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenStreamEvent]:
        """Stream generation events from llama.cpp server's /completion endpoint.

        Parses Server-Sent Events (SSE). Emits:
        1. START event at request dispatch.
        2. CHUNK (or TOKEN if token IDs are present) events as text fragments arrive.
        3. COMPLETION event containing llama.cpp server timings, or ERROR on failure.
        """
        client = self.get_client()
        prompt_text = resolve_prompt_text(request.workload)

        # Build llama.cpp request payload
        payload: dict[str, Any] = {
            "prompt": prompt_text,
            "stream": True,
        }
        if request.workload.output_tokens is not None:
            payload["n_predict"] = request.workload.output_tokens
        if request.workload.temperature is not None:
            payload["temperature"] = request.workload.temperature
        if request.workload.top_p is not None:
            payload["top_p"] = request.workload.top_p
        if request.workload.stop_sequences:
            payload["stop"] = request.workload.stop_sequences
        if request.workload.seed is not None:
            payload["seed"] = request.workload.seed

        # Merge engine config extra_args
        for k, v in request.engine_config.extra_args.items():
            payload[k] = v

        # 1. Emit START event
        start_time_ns = time.perf_counter_ns()
        seq_index = 0
        yield TokenStreamEvent(
            request_id=request.request_id,
            sequence_index=seq_index,
            event_type=TokenEventType.START,
            timestamp_ns=start_time_ns,
            delta_ns=None,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        )
        seq_index += 1
        prev_time_ns = start_time_ns
        token_count = 0
        completed = False

        try:
            async with client.stream("POST", "/completion", json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_text = body.decode("utf-8", errors="replace")
                    raise EngineExecutionError(
                        f"llama.cpp server returned HTTP {response.status_code}: {error_text}"
                    )

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        # Skip empty lines and SSE comment / keepalive lines
                        continue

                    if not line.startswith("data:"):
                        continue

                    line_time_ns = time.perf_counter_ns()
                    delta_ns = max(0, line_time_ns - prev_time_ns)
                    data_str = line[5:].strip()

                    # Handle empty data line (SSE keepalive/ping)
                    if not data_str:
                        continue

                    # Handle [DONE] marker
                    if data_str == "[DONE]":
                        if not completed:
                            yield TokenStreamEvent(
                                request_id=request.request_id,
                                sequence_index=seq_index,
                                event_type=TokenEventType.COMPLETION,
                                timestamp_ns=line_time_ns,
                                delta_ns=delta_ns,
                                clock_source=ClockSource.MONOTONIC_REAL,
                                execution_mode=ExecutionMode.REAL,
                                is_terminal=True,
                                termination_reason=StreamTerminationReason.FINISHED,
                            )
                            completed = True
                        return

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError as err:
                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=seq_index,
                            event_type=TokenEventType.ERROR,
                            timestamp_ns=line_time_ns,
                            delta_ns=delta_ns,
                            clock_source=ClockSource.MONOTONIC_REAL,
                            execution_mode=ExecutionMode.REAL,
                            is_terminal=True,
                            termination_reason=StreamTerminationReason.ERROR,
                            error_message=f"Malformed SSE JSON from llama.cpp: {err} in '{data_str}'",
                        )
                        completed = True
                        return

                    # Handle server error payload
                    if "error" in data:
                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=seq_index,
                            event_type=TokenEventType.ERROR,
                            timestamp_ns=line_time_ns,
                            delta_ns=delta_ns,
                            clock_source=ClockSource.MONOTONIC_REAL,
                            execution_mode=ExecutionMode.REAL,
                            is_terminal=True,
                            termination_reason=StreamTerminationReason.ERROR,
                            error_message=f"llama.cpp stream error: {data['error']}",
                        )
                        completed = True
                        return

                    content = data.get("content", "")
                    tokens_data = data.get("tokens")
                    tok_id = data.get("token_id")

                    has_tokens_list = isinstance(tokens_data, list)
                    is_multi_token = has_tokens_list and len(tokens_data) > 1

                    # Protocol guarantee check:
                    # In standard HTTP streaming, SSE frames are text chunks without individual token timing guarantees.
                    # Only if verified_token_stream is True in engine config does the adapter emit TOKEN events.
                    verified_stream = bool(
                        request.engine_config.extra_args.get("verified_token_stream", False)
                    )

                    single_tok_id = (
                        tok_id
                        if tok_id is not None
                        else (tokens_data[0] if (has_tokens_list and len(tokens_data) == 1) else None)
                    )
                    is_discrete_token = (
                        verified_stream and (not is_multi_token) and (single_tok_id is not None)
                    )

                    if is_discrete_token:
                        is_first = token_count == 0
                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=seq_index,
                            event_type=TokenEventType.TOKEN,
                            token_index=token_count,
                            token_id=single_tok_id,
                            token_text=content,
                            timestamp_ns=line_time_ns,
                            delta_ns=delta_ns,
                            clock_source=ClockSource.MONOTONIC_REAL,
                            execution_mode=ExecutionMode.REAL,
                            is_first_token=is_first,
                        )
                        token_count += 1
                        seq_index += 1
                        prev_time_ns = line_time_ns
                    elif content or has_tokens_list or tok_id is not None:
                        # Emitted as CHUNK: multi-token array, unverified token array, or text chunk
                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=seq_index,
                            event_type=TokenEventType.CHUNK,
                            token_text=content,
                            timestamp_ns=line_time_ns,
                            delta_ns=delta_ns,
                            clock_source=ClockSource.MONOTONIC_REAL,
                            execution_mode=ExecutionMode.REAL,
                        )
                        seq_index += 1
                        prev_time_ns = line_time_ns

                    # Check for completion
                    if data.get("stop") is True:
                        timings = data.get("timings", {})
                        prompt_n = timings.get("prompt_n", data.get("tokens_evaluated"))
                        prompt_ms = timings.get("prompt_ms")
                        predicted_n = timings.get(
                            "predicted_n",
                            data.get("tokens_predicted", data.get("n_decoded")),
                        )
                        predicted_ms = timings.get("predicted_ms")
                        predicted_tps = timings.get("predicted_per_second")

                        stats = ServerReportedStats(
                            prompt_tokens=prompt_n,
                            output_tokens=predicted_n,
                            prompt_duration_ms=prompt_ms,
                            eval_duration_ms=predicted_ms,
                            eval_tokens_per_second=predicted_tps,
                            raw_stats=data,
                        )

                        if data.get("stopped_limit") or (
                            request.workload.output_tokens > 0
                            and predicted_n == request.workload.output_tokens
                        ):
                            reason = StreamTerminationReason.MAX_TOKENS
                        else:
                            reason = StreamTerminationReason.FINISHED

                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=seq_index,
                            event_type=TokenEventType.COMPLETION,
                            timestamp_ns=line_time_ns,
                            delta_ns=max(0, line_time_ns - prev_time_ns),
                            clock_source=ClockSource.MONOTONIC_REAL,
                            execution_mode=ExecutionMode.REAL,
                            is_terminal=True,
                            termination_reason=reason,
                            server_stats=stats,
                        )
                        completed = True
                        return

        except (asyncio.CancelledError, StreamCancelledError, TimeoutError):
            if not completed:
                term_time_ns = time.perf_counter_ns()
                yield TokenStreamEvent(
                    request_id=request.request_id,
                    sequence_index=seq_index,
                    event_type=TokenEventType.COMPLETION,
                    timestamp_ns=term_time_ns,
                    delta_ns=max(0, term_time_ns - prev_time_ns),
                    clock_source=ClockSource.MONOTONIC_REAL,
                    execution_mode=ExecutionMode.REAL,
                    is_terminal=True,
                    termination_reason=StreamTerminationReason.CANCELLED,
                )
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            term_time_ns = time.perf_counter_ns()
            yield TokenStreamEvent(
                request_id=request.request_id,
                sequence_index=seq_index,
                event_type=TokenEventType.ERROR,
                timestamp_ns=term_time_ns,
                delta_ns=max(0, term_time_ns - prev_time_ns),
                clock_source=ClockSource.MONOTONIC_REAL,
                execution_mode=ExecutionMode.REAL,
                is_terminal=True,
                termination_reason=StreamTerminationReason.ERROR,
                error_message=f"Connection to llama.cpp server failed: {exc}",
            )
            raise EngineConnectionError(
                f"Connection to llama.cpp server at {self._base_url} failed: {exc}"
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            term_time_ns = time.perf_counter_ns()
            yield TokenStreamEvent(
                request_id=request.request_id,
                sequence_index=seq_index,
                event_type=TokenEventType.ERROR,
                timestamp_ns=term_time_ns,
                delta_ns=max(0, term_time_ns - prev_time_ns),
                clock_source=ClockSource.MONOTONIC_REAL,
                execution_mode=ExecutionMode.REAL,
                is_terminal=True,
                termination_reason=StreamTerminationReason.ERROR,
                error_message=f"llama.cpp server read timeout: {exc}",
            )
            raise EngineTimeoutError(
                f"llama.cpp server read timeout after {self._timeout_seconds}s: {exc}"
            ) from exc
        except Exception as exc:
            if not completed:
                term_time_ns = time.perf_counter_ns()
                yield TokenStreamEvent(
                    request_id=request.request_id,
                    sequence_index=seq_index,
                    event_type=TokenEventType.ERROR,
                    timestamp_ns=term_time_ns,
                    delta_ns=max(0, term_time_ns - prev_time_ns),
                    clock_source=ClockSource.MONOTONIC_REAL,
                    execution_mode=ExecutionMode.REAL,
                    is_terminal=True,
                    termination_reason=StreamTerminationReason.ERROR,
                    error_message=str(exc).strip() or f"{type(exc).__name__}: Stream error",
                )
            if not isinstance(exc, EngineExecutionError):
                raise EngineExecutionError(f"llama.cpp stream error: {exc}") from exc
            raise

        if not completed:
            term_time_ns = time.perf_counter_ns()
            yield TokenStreamEvent(
                request_id=request.request_id,
                sequence_index=seq_index,
                event_type=TokenEventType.ERROR,
                timestamp_ns=term_time_ns,
                delta_ns=max(0, term_time_ns - prev_time_ns),
                clock_source=ClockSource.MONOTONIC_REAL,
                execution_mode=ExecutionMode.REAL,
                is_terminal=True,
                termination_reason=StreamTerminationReason.ERROR,
                error_message="llama.cpp stream terminated prematurely without stop marker",
            )
