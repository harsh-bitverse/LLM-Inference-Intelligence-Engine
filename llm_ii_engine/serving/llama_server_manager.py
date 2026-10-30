"""Lifecycle manager for session-owned llama.cpp server (llama-server.exe).

Provides safe process management, health checking, slot telemetry querying,
and guaranteed fail-safe cleanup so that temporary test servers never leak.
"""

import json
import logging
import subprocess
import time
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from llm_ii_engine.serving.capabilities import (
    LLAMA_SERVER_PATH,
    ServingCapabilityInspector,
)

logger = logging.getLogger(__name__)


class LlamaServerProcess:
    """Manages a single session-owned instance of llama-server.exe."""

    def __init__(
        self,
        model_id: str | None = None,
        slots: int = 2,
        continuous_batching: bool = True,
        threads: int = 8,
        port: int = 18080,
        host: str = "127.0.0.1",
        context_size: int = 2048,
        num_ctx: int | None = None,
        binary_path: str | None = None,
        model_path: str | None = None,
    ):
        self.model_id = model_id or "qwen2.5:0.5b"
        self.slots = slots
        self.continuous_batching = continuous_batching
        self.threads = threads
        self.port = port
        self.host = host
        self.context_size = num_ctx if num_ctx is not None else context_size
        self.binary_path = binary_path or str(LLAMA_SERVER_PATH)
        self.model_path = model_path
        self.process: subprocess.Popen | None = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_command(self) -> list[str]:
        """Construct the CLI argument list for llama-server.exe."""
        model_path_str = self.model_path
        if not model_path_str:
            resolved = ServingCapabilityInspector.get_model_blob_path(self.model_id)
            model_path_str = str(resolved) if resolved else "model.gguf"

        cmd = [
            self.binary_path,
            "-m",
            model_path_str,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "-np",
            str(self.slots),
            "-c",
            str(self.context_size),
            "-t",
            str(self.threads),
        ]
        if self.continuous_batching:
            cmd.append("-cb")
        else:
            cmd.append("-nocb")
        return cmd

    def start(self, timeout_s: float = 20.0) -> None:
        """Launch llama-server.exe subprocess and wait until endpoint is responsive."""
        if not Path(self.binary_path).exists() and not ServingCapabilityInspector.is_llama_server_available():
            raise RuntimeError(f"llama-server executable not found at '{self.binary_path}'")

        if not self.model_path:
            model_path = ServingCapabilityInspector.get_model_blob_path(self.model_id)
            if not model_path:
                raise FileNotFoundError(f"GGUF model blob not found for model '{self.model_id}'")

        cmd = self.build_command()

        logger.info(f"Starting llama-server on port {self.port} with {self.slots} slots (cont_batching={self.continuous_batching})...")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Poll for readiness
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_s:
            if self.process.poll() is not None:
                out = self.process.stdout.read() if self.process.stdout else ""
                raise RuntimeError(f"llama-server process exited prematurely with code {self.process.returncode}:\n{out}")
            if self.is_healthy():
                logger.info(f"llama-server is ready and healthy at {self.base_url}")
                return
            time.sleep(0.3)

        self.stop()
        raise TimeoutError(f"llama-server failed to respond within {timeout_s} seconds.")

    def is_healthy(self) -> bool:
        """Query /slots or /health to verify server responsiveness."""
        try:
            req = urllib.request.Request(f"{self.base_url}/slots")
            with self._opener.open(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def query_slots(self) -> list[dict[str, Any]]:
        """Query /slots endpoint to inspect slot execution state."""
        try:
            req = urllib.request.Request(f"{self.base_url}/slots")
            with self._opener.open(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to query /slots: {e}")
            return []

    def stop(self) -> None:
        """Terminate the server process cleanly."""
        if self.process and self.process.poll() is None:
            logger.info("Terminating session-owned llama-server...")
            try:
                self.process.terminate()
                self.process.wait(timeout=3.0)
            except Exception:
                with suppress(Exception):
                    self.process.kill()
                    self.process.wait(timeout=2.0)
            finally:
                if self.process.stdout:
                    with suppress(Exception):
                        self.process.stdout.close()
                self.process = None


@contextmanager
def LlamaServerContext(
    model_id: str,
    slots: int = 2,
    continuous_batching: bool = True,
    threads: int = 8,
    port: int = 18080,
    context_size: int = 2048,
) -> Generator[LlamaServerProcess, None, None]:
    """Context manager ensuring fail-safe lifecycle for llama-server."""
    server = LlamaServerProcess(
        model_id=model_id,
        slots=slots,
        continuous_batching=continuous_batching,
        threads=threads,
        port=port,
        context_size=context_size,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
