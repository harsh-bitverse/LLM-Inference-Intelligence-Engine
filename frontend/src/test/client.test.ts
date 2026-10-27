import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EngineApiClient, ApiError } from '../api/client';
import { HealthResponse, TokenStreamEvent } from '../api/types';

describe('EngineApiClient', () => {
  let client: EngineApiClient;

  beforeEach(() => {
    client = new EngineApiClient('http://localhost:8000');
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('normalizes API errors with structured ErrorDetail', async () => {
    const errorBody = {
      error: 'RESOURCE_LIMIT_EXCEEDED',
      message: 'Model memory requires 32 GB, but GPU capacity is 24 GB',
      detail: 'Model memory requires 32 GB, but GPU capacity is 24 GB',
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 422,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => errorBody,
    } as Response);

    await expect(client.checkHealth()).rejects.toThrowError(ApiError);

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 422,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => errorBody,
    } as Response);

    try {
      await client.checkHealth();
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.statusCode).toBe(422);
      expect(apiErr.errorDetail).toEqual(errorBody);
    }
  });

  it('parses successful GET /health correctly', async () => {
    const mockHealth: HealthResponse = {
      status: 'ok',
      database: 'connected',
      version: '1.0.0',
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => mockHealth,
    } as Response);

    const res = await client.checkHealth();
    expect(res.status).toBe('ok');
    expect(res.version).toBe('1.0.0');
    expect(res.database).toBe('connected');
  });

  it('streams SSE token events and handles chunk framing', async () => {
    const event1: TokenStreamEvent = {
      request_id: 'req-1',
      sequence_index: 0,
      event_type: 'token',
      token_index: 0,
      token_text: 'Hello',
      timestamp_ns: 1000,
      is_first_token: true,
      is_terminal: false,
      clock_source: 'MONOTONIC_REAL',
      execution_mode: 'SIMULATED',
    };

    const event2: TokenStreamEvent = {
      request_id: 'req-1',
      sequence_index: 1,
      event_type: 'token',
      token_index: 1,
      token_text: ' world',
      timestamp_ns: 2000,
      is_first_token: false,
      is_terminal: true,
      clock_source: 'MONOTONIC_REAL',
      execution_mode: 'SIMULATED',
    };

    const streamChunks = [
      `data: ${JSON.stringify(event1)}\n\n`,
      `data: ${JSON.stringify(event2)}\n\n`,
    ];

    const encoder = new TextEncoder();
    let chunkIndex = 0;

    const mockReadableStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (chunkIndex < streamChunks.length) {
          controller.enqueue(encoder.encode(streamChunks[chunkIndex]));
          chunkIndex++;
        } else {
          controller.close();
        }
      },
    });

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'text/event-stream' }),
      body: mockReadableStream,
    } as unknown as Response);

    const receivedEvents: TokenStreamEvent[] = [];
    let completed = false;

    await new Promise<void>((resolve) => {
      client.subscribeJobStream(
        'job-123',
        (evt) => {
          receivedEvents.push(evt);
        },
        undefined,
        () => {
          completed = true;
          resolve();
        }
      );
    });

    expect(receivedEvents).toHaveLength(2);
    expect(receivedEvents[0]?.token_text).toBe('Hello');
    expect(receivedEvents[1]?.token_text).toBe(' world');
    expect(completed).toBe(true);
  });

  it('normalizes CRLF (\\r\\n) framing and multi-line data payloads in SSE stream', async () => {
    const event: TokenStreamEvent = {
      request_id: 'req-crlf',
      sequence_index: 0,
      event_type: 'token',
      token_index: 0,
      token_text: 'CRLF-Token',
      timestamp_ns: 5000,
      is_first_token: true,
      is_terminal: true,
      clock_source: 'MONOTONIC_REAL',
      execution_mode: 'SIMULATED',
    };

    // Construct stream chunk with CRLF and multi-line data fields
    const jsonStr = JSON.stringify(event, null, 2);
    const crlfChunk = jsonStr
      .split('\n')
      .map((line) => `data: ${line}\r\n`)
      .join('') + '\r\n';

    const encoder = new TextEncoder();
    let sent = false;

    const mockReadableStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (!sent) {
          controller.enqueue(encoder.encode(crlfChunk));
          sent = true;
        } else {
          controller.close();
        }
      },
    });

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'text/event-stream' }),
      body: mockReadableStream,
    } as unknown as Response);

    const receivedEvents: TokenStreamEvent[] = [];
    let completed = false;

    await new Promise<void>((resolve) => {
      client.subscribeJobStream(
        'job-crlf',
        (evt) => {
          receivedEvents.push(evt);
        },
        undefined,
        () => {
          completed = true;
          resolve();
        }
      );
    });

    expect(receivedEvents).toHaveLength(1);
    expect(receivedEvents[0]?.token_text).toBe('CRLF-Token');
    expect(completed).toBe(true);
  });

  it('retrieves hardware catalog specifications from GET /api/v1/hardware/catalog', async () => {
    const mockCatalog = [
      {
        catalog_id: 'nvidia-h100-sxm',
        name: 'NVIDIA H100 SXM5 80GB',
        architecture: 'Hopper',
        form_factor: 'SXM5',
        memory_capacity_gb: 80.0,
        memory_type: 'HBM3',
        theoretical_memory_bandwidth_gbps: 3350.0,
        compute: { fp16_dense_tflops: 989.4 },
        source_reference: 'NVIDIA H100 Datasheet',
        caveats: [],
        provenance: 'THEORETICAL' as const,
      },
    ];

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => mockCatalog,
    } as Response);

    const catalog = await client.getHardwareCatalog();
    expect(catalog).toHaveLength(1);
    expect(catalog[0]?.catalog_id).toBe('nvidia-h100-sxm');
    expect(catalog[0]?.theoretical_memory_bandwidth_gbps).toBe(3350.0);
    expect(catalog[0]?.compute.fp16_dense_tflops).toBe(989.4);
  });

  it('retrieves optimization run by ID from GET /api/v1/optimize/{id}', async () => {
    const mockOptDetail = {
      request: {
        workload: { prompt_tokens: 128, output_tokens: 64 },
        search_space: {
          models: [{ model_id: 'meta-llama/Llama-3-8B' }],
          quantizations: ['fp16'],
          concurrencies: [1],
          engines: ['mock'],
        },
        objectives: ['minimize_ttft'],
      },
      result: {
        request_id: 'opt-test-123',
        summary: 'Pareto optimization completed',
        total_candidates_generated: 4,
        total_candidates_evaluated: 4,
        feasible_candidates_count: 4,
        infeasible_candidates_count: 0,
        unknown_feasibility_count: 0,
        pareto_frontier: [],
        rejections_by_reason: {},
        all_evaluations: [],
        has_incomparable_evidence: false,
        evidence_caveats: [],
      },
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => mockOptDetail,
    } as Response);

    const detail = await client.getOptimizationById('opt-test-123');
    expect(detail.result.request_id).toBe('opt-test-123');
    expect(detail.result.feasible_candidates_count).toBe(4);
    expect(detail.request.workload.prompt_tokens).toBe(128);
  });
});
