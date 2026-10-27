"""Deterministic Mock Inference Engine Adapter.

Simulates prefill (TTFT) and decode (ITL) streaming token generation
with virtual nanosecond advancement and explicit simulated execution tagging.
Enables 100% deterministic test execution on CPU without GPU drivers or model weights.
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.adapters.base import BaseInferenceAdapter
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
    StreamCancelledError,
)
from llm_ii_engine.core.schemas import (
    EngineCapabilities,
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    TokenStreamEvent,
)


class MockAdapterConfig(BaseModel):
    """Configuration governing the simulation behavior of MockEngineAdapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefill_delay_ns: int = Field(
        default=50_000_000,
        ge=0,
        description="Virtual or physical delay for first token (prefill) in nanoseconds",
    )
    inter_token_delay_ns: int = Field(
        default=10_000_000,
        ge=0,
        description="Virtual or physical delay between decode tokens in nanoseconds",
    )
    jitter_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of random jitter added to delays (0.0 = deterministic)",
    )
    seed: int | None = Field(
        default=42, description="Pseudorandom generator seed for reproducible jitter"
    )
    virtual_time_mode: bool = Field(
        default=True,
        description="When True, advances synthetic timestamps with zero real sleep for fast, deterministic testing",
    )
    initial_timestamp_ns: int = Field(
        default=1_000_000_000,
        ge=0,
        description="Base starting timestamp for virtual clock mode",
    )
    simulated_memory_mb: float = Field(
        default=1024.0,
        ge=0.0,
        description="Static memory usage reported by get_memory_usage_mb()",
    )
    token_prefix: str = Field(
        default="sim_tok_",
        min_length=1,
        description="Prefix applied to synthetic token text",
    )
    fail_on_initialize: bool = Field(
        default=False,
        description="Simulate an initialization failure for testing error handling",
    )
    fail_on_health_check: bool = Field(
        default=False,
        description="Simulate a failed health check for testing connectivity handling",
    )
    error_at_token_index: int | None = Field(
        default=None,
        ge=0,
        description="If set, triggers a simulated failure at the given token index",
    )
    error_delivery_mode: str = Field(
        default="exception",
        pattern="^(exception|event)$",
        description="How to deliver injected error: 'exception' raises EngineExecutionError, 'event' yields ERROR event",
    )


class MockEngineAdapter(BaseInferenceAdapter):
    """Deterministic, mock LLM inference engine adapter.

    Emits structured token events labeled with ExecutionMode.SIMULATED and
    ClockSource.SYNTHETIC_VIRTUAL (in virtual time mode).
    Outputs synthetic tokens prefixed with 'sim_tok_' to prevent any confusion
    with real model inference.
    """

    def __init__(
        self,
        config: MockAdapterConfig | None = None,
        name: str = "MockEngineAdapter",
    ) -> None:
        super().__init__(
            name=name,
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.SIMULATED,
        )
        self._config = config or MockAdapterConfig()

    @property
    def config(self) -> MockAdapterConfig:
        return self._config

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.SIMULATED,
            supported_quantizations=[
                QuantizationType.NONE,
                QuantizationType.FP16,
                QuantizationType.BF16,
                QuantizationType.INT8,
                QuantizationType.INT4,
                QuantizationType.GGUF_Q4_K_M,
            ],
            supports_streaming=True,
            supports_gpu_memory_query=True,
            supports_token_ids=True,
            max_context_window=32768,
        )

    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool:
        if self._config.fail_on_initialize:
            self._is_initialized = False
            raise EngineConnectionError("Simulated initialization failure in MockEngineAdapter")
        self._is_initialized = True
        return True

    async def health_check(self) -> bool:
        return not self._config.fail_on_health_check

    async def get_memory_usage_mb(self) -> float | None:
        return self._config.simulated_memory_mb

    async def teardown(self) -> None:
        await super().teardown()

    def _compute_delay(self, base_delay_ns: int, rng: random.Random) -> int:
        if self._config.jitter_ratio <= 0.0 or base_delay_ns <= 0:
            return base_delay_ns
        jitter_span = int(base_delay_ns * self._config.jitter_ratio)
        offset = rng.randint(-jitter_span, jitter_span)
        return max(1, base_delay_ns + offset)

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenStreamEvent]:
        """Stream synthetic token events according to the request workload.

        Yields:
            TokenStreamEvent sequence: START -> [TOKEN * N] -> COMPLETION (or ERROR).

        Raises:
            EngineExecutionError: When simulated failure is configured as an exception.
            StreamCancelledError: When the generation task is cancelled.
        """
        # Determine pseudorandom generator seed for reproducibility
        seed = request.workload.seed if request.workload.seed is not None else self._config.seed
        rng = random.Random(seed)

        virtual_mode = self._config.virtual_time_mode
        clock_source = ClockSource.SYNTHETIC_VIRTUAL if virtual_mode else ClockSource.MONOTONIC_REAL

        # Base timestamp initialization
        current_time_ns = (
            self._config.initial_timestamp_ns if virtual_mode else time.perf_counter_ns()
        )
        sequence_idx = 0

        try:
            # 1. Emit START event
            yield TokenStreamEvent(
                request_id=request.request_id,
                sequence_index=sequence_idx,
                event_type=TokenEventType.START,
                token_index=None,
                token_text=None,
                token_id=None,
                timestamp_ns=current_time_ns,
                delta_ns=None,
                clock_source=clock_source,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=False,
                is_terminal=False,
                termination_reason=None,
                error_message=None,
            )
            sequence_idx += 1

            # 2. Prefill phase delay
            prefill_delay = self._compute_delay(self._config.prefill_delay_ns, rng)
            if virtual_mode:
                current_time_ns += prefill_delay
            else:
                await asyncio.sleep(prefill_delay / 1e9)
                current_time_ns = time.perf_counter_ns()

            # 3. Check for zero-token edge case (prefill-only workload)
            if request.workload.output_tokens == 0:
                yield TokenStreamEvent(
                    request_id=request.request_id,
                    sequence_index=sequence_idx,
                    event_type=TokenEventType.COMPLETION,
                    token_index=None,
                    token_text="",
                    token_id=None,
                    timestamp_ns=current_time_ns,
                    delta_ns=prefill_delay,
                    clock_source=clock_source,
                    execution_mode=ExecutionMode.SIMULATED,
                    is_first_token=False,
                    is_terminal=True,
                    termination_reason=StreamTerminationReason.MAX_TOKENS,
                    error_message=None,
                )
                return

            # 4. Generate tokens sequentially
            last_timestamp_ns = current_time_ns - prefill_delay
            for tok_i in range(request.workload.output_tokens):
                # Check for injected error condition
                if (
                    self._config.error_at_token_index is not None
                    and tok_i == self._config.error_at_token_index
                ):
                    if self._config.error_delivery_mode == "event":
                        yield TokenStreamEvent(
                            request_id=request.request_id,
                            sequence_index=sequence_idx,
                            event_type=TokenEventType.ERROR,
                            token_index=tok_i,
                            token_text=None,
                            token_id=None,
                            timestamp_ns=current_time_ns,
                            delta_ns=current_time_ns - last_timestamp_ns,
                            clock_source=clock_source,
                            execution_mode=ExecutionMode.SIMULATED,
                            is_first_token=False,
                            is_terminal=True,
                            termination_reason=StreamTerminationReason.ERROR,
                            error_message=f"Simulated injected failure at token index {tok_i}",
                        )
                        return
                    else:
                        raise EngineExecutionError(
                            f"Simulated injected exception in MockEngineAdapter at token index {tok_i}"
                        )

                # Decode phase delay (for tokens after the first token)
                if tok_i > 0:
                    itl_delay = self._compute_delay(self._config.inter_token_delay_ns, rng)
                    if virtual_mode:
                        current_time_ns += itl_delay
                    else:
                        await asyncio.sleep(itl_delay / 1e9)
                        current_time_ns = time.perf_counter_ns()
                    delta_ns = itl_delay
                else:
                    # First token: delta represents the TTFT (prefill delay)
                    delta_ns = prefill_delay

                token_text = f"{self._config.token_prefix}{tok_i}"
                token_id = 1000 + tok_i

                yield TokenStreamEvent(
                    request_id=request.request_id,
                    sequence_index=sequence_idx,
                    event_type=TokenEventType.TOKEN,
                    token_index=tok_i,
                    token_text=token_text,
                    token_id=token_id,
                    timestamp_ns=current_time_ns,
                    delta_ns=delta_ns,
                    clock_source=clock_source,
                    execution_mode=ExecutionMode.SIMULATED,
                    is_first_token=(tok_i == 0),
                    is_terminal=False,
                    termination_reason=None,
                    error_message=None,
                )
                sequence_idx += 1
                last_timestamp_ns = current_time_ns

            # 5. Emit terminal COMPLETION event
            # A minimal virtual wrap-up interval ensures monotonic progression
            wrapup_delay_ns = 1_000 if virtual_mode else 0
            current_time_ns += wrapup_delay_ns

            yield TokenStreamEvent(
                request_id=request.request_id,
                sequence_index=sequence_idx,
                event_type=TokenEventType.COMPLETION,
                token_index=request.workload.output_tokens,
                token_text="",
                token_id=None,
                timestamp_ns=current_time_ns,
                delta_ns=wrapup_delay_ns,
                clock_source=clock_source,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=False,
                is_terminal=True,
                termination_reason=StreamTerminationReason.FINISHED,
                error_message=None,
            )

        except asyncio.CancelledError:
            raise StreamCancelledError(
                f"Stream generation for request {request.request_id} was cancelled"
            )
