"""Ollama HTTP streaming inference engine adapter.

Interfaces with Ollama's /api/generate endpoint, streaming incremental JSON lines,
measuring client-observed chunk arrivals using monotonic clocks, and capturing
server-side evaluation statistics without confusing text chunks with discrete tokens.
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
    ModelNotFoundError,
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


class OllamaAdapter(BaseHttpEngineAdapter):
    """Production-conscious HTTP streaming adapter for Ollama."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str = "OllamaAdapter",
    ) -> None:
        """Initialize Ollama adapter.

        Args:
            base_url: Ollama server base URL (defaults to http://localhost:11434).
            timeout_seconds: Overall request/stream timeout.
            connect_timeout_seconds: Connection timeout.
            client: Optional injected HTTP client.
            transport: Optional injected HTTP transport.
            name: Adapter identifier.
        """
        super().__init__(
            name=name,
            engine_type=EngineType.OLLAMA,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            client=client,
            transport=transport,
        )
        self._model_spec: ModelSpec | None = None
        self._engine_config: EngineConfig | None = None

    def get_capabilities(self) -> EngineCapabilities:
        """Return Ollama capabilities."""
        return EngineCapabilities(
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            supported_quantizations=[
                QuantizationType.NONE,
                QuantizationType.GGUF_Q4_K_M,
                QuantizationType.GGUF_Q8_0,
                QuantizationType.FP16,
            ],
            supports_streaming=True,
            supports_gpu_memory_query=False,
            supports_token_ids=False,
            supports_true_token_itl=False,  # Ollama streams text chunks, not true tokens
            supports_server_metrics=True,  # Ollama provides prompt_eval_duration and eval_duration
            max_context_window=None,
        )

    async def health_check(self) -> bool:
        """Verify Ollama server responsiveness via /api/version."""
        client = self.get_client()
        try:
            resp = await client.get("/api/version")
            return resp.status_code == 200
        except Exception:
            return False

    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool:
        """Initialize adapter and verify model existence on Ollama server without pulling.

        Args:
            model_spec: Target model specification.
            engine_config: Engine configuration parameters.

        Raises:
            EngineConnectionError: If Ollama server is unreachable.
            ModelNotFoundError: If requested model is not already installed on Ollama server.
        """
        client = self.get_client()

        # 1. Health & version check
        try:
            version_resp = await client.get("/api/version")
            if version_resp.status_code != 200:
                raise EngineConnectionError(
                    f"Ollama server at {self._base_url} returned HTTP {version_resp.status_code} on version check"
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self._map_http_error(exc, "/api/version")
        except Exception as exc:
            if not isinstance(exc, EngineConnectionError):
                raise EngineConnectionError(
                    f"Failed to communicate with Ollama server at {self._base_url}: {exc}"
                ) from exc
            raise

        # 2. Check model existence via /api/tags
        skip_check = engine_config.extra_args.get("skip_model_check", False)
        if not skip_check:
            try:
                tags_resp = await client.get("/api/tags")
                if tags_resp.status_code == 200:
                    data = tags_resp.json()
                    models_list = [m.get("name", "") for m in data.get("models", [])]
                    target = model_spec.model_id
                    # Match exact name, with :latest tag, or base name
                    is_present = any(
                        target == m
                        or f"{target}:latest" == m
                        or target.split(":")[0] == m.split(":")[0]
                        for m in models_list
                    )
                    if not is_present:
                        raise ModelNotFoundError(
                            f"Model '{target}' is not installed on Ollama server at {self._base_url}. "
                            f"Installed models: {models_list}. "
                            f"Model downloading is disabled; please run 'ollama pull {target}' manually before benchmarking."
                        )
            except ModelNotFoundError:
                raise
            except Exception as exc:
                if isinstance(exc, (EngineConnectionError, EngineTimeoutError)):
                    raise
                # Non-fatal if /api/tags is absent on a customized server

        self._model_spec = model_spec
        self._engine_config = engine_config
        self._is_initialized = True
        return True

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenStreamEvent]:
        """Stream generation events from Ollama's /api/generate endpoint.

        Emits:
        1. START event at request dispatch.
        2. CHUNK events as incremental text deltas arrive from the stream.
        3. COMPLETION event containing Ollama server evaluation statistics, or ERROR on failure.
        """
        client = self.get_client()
        prompt_text = resolve_prompt_text(request.workload)

        # Build Ollama options dictionary
        options: dict[str, Any] = {}
        if request.workload.output_tokens is not None:
            options["num_predict"] = request.workload.output_tokens
        if request.workload.temperature is not None:
            options["temperature"] = request.workload.temperature
        if request.workload.top_p is not None:
            options["top_p"] = request.workload.top_p
        if request.workload.stop_sequences:
            options["stop"] = request.workload.stop_sequences
        if request.workload.seed is not None:
            options["seed"] = request.workload.seed

        # Merge engine config extra_args
        for k, v in request.engine_config.extra_args.items():
            if k != "skip_model_check":
                options[k] = v

        payload = {
            "model": request.model_spec.model_id,
            "prompt": prompt_text,
            "stream": True,
            "options": options,
        }

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
        completed = False

        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_text = body.decode("utf-8", errors="replace")
                    if response.status_code == 404:
                        raise ModelNotFoundError(
                            f"Ollama model '{request.model_spec.model_id}' not found: {error_text}"
                        )
                    raise EngineExecutionError(
                        f"Ollama server returned HTTP {response.status_code}: {error_text}"
                    )

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    line_time_ns = time.perf_counter_ns()
                    delta_ns = max(0, line_time_ns - prev_time_ns)

                    try:
                        data = json.loads(line)
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
                            error_message=f"Malformed JSON from Ollama stream: {err} in line '{line}'",
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
                            error_message=f"Ollama stream error: {data['error']}",
                        )
                        completed = True
                        return

                    chunk_text = data.get("response", "")
                    if chunk_text:
                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=seq_index,
                            event_type=TokenEventType.CHUNK,
                            token_text=chunk_text,
                            timestamp_ns=line_time_ns,
                            delta_ns=delta_ns,
                            clock_source=ClockSource.MONOTONIC_REAL,
                            execution_mode=ExecutionMode.REAL,
                        )
                        seq_index += 1
                        prev_time_ns = line_time_ns

                    # Check for completion
                    if data.get("done") is True:
                        prompt_eval_count = data.get("prompt_eval_count")
                        eval_count = data.get("eval_count")
                        prompt_eval_cached_count = data.get("prompt_eval_cached_count")
                        prompt_eval_duration = data.get("prompt_eval_duration")
                        eval_duration = data.get("eval_duration")
                        total_duration = data.get("total_duration")
                        load_duration = data.get("load_duration")

                        prompt_ms = (
                            prompt_eval_duration / 1_000_000.0
                            if prompt_eval_duration is not None
                            else None
                        )
                        eval_ms = eval_duration / 1_000_000.0 if eval_duration is not None else None
                        total_ms = (
                            total_duration / 1_000_000.0 if total_duration is not None else None
                        )
                        load_ms = (
                            load_duration / 1_000_000.0 if load_duration is not None else None
                        )
                        eval_tps = (
                            (eval_count / (eval_duration / 1e9))
                            if (eval_count and eval_duration and eval_duration > 0)
                            else None
                        )

                        stats = ServerReportedStats(
                            prompt_tokens=prompt_eval_count,
                            output_tokens=eval_count,
                            prompt_duration_ms=prompt_ms,
                            eval_duration_ms=eval_ms,
                            total_duration_ms=total_ms,
                            eval_tokens_per_second=eval_tps,
                            prompt_eval_cached_count=prompt_eval_cached_count,
                            load_duration_ms=load_ms,
                            raw_stats=data,
                        )

                        done_reason = data.get("done_reason")
                        if done_reason == "length" or (
                            request.workload.output_tokens > 0
                            and eval_count == request.workload.output_tokens
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
                error_message=f"Connection to Ollama server failed: {exc}",
            )
            raise EngineConnectionError(
                f"Connection to Ollama server at {self._base_url} failed: {exc}"
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
                error_message=f"Ollama server read timeout: {exc}",
            )
            raise EngineTimeoutError(
                f"Ollama server read timeout after {self._timeout_seconds}s: {exc}"
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
                raise EngineExecutionError(f"Ollama stream error: {exc}") from exc
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
                error_message="Ollama stream terminated prematurely without done=true marker",
            )
