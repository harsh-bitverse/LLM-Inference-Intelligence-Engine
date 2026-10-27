"""Security, CORS configuration, path sanitization, and hardware catalog regression tests."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.api.errors import sanitize_storage_error_message
from llm_ii_engine.discovery.catalog import list_gpu_specs
from llm_ii_engine.storage.database import DatabaseManager


@pytest.fixture
def temp_db_url(tmp_path: Path) -> str:
    db_file = tmp_path / "test_security.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def db_manager(temp_db_url: str) -> DatabaseManager:
    mgr = DatabaseManager(database_url=temp_db_url)
    mgr.init_db()
    return mgr


@pytest.fixture
def app_with_mgr(db_manager: DatabaseManager):
    return create_app(db_manager=db_manager)


@pytest.fixture
async def client(app_with_mgr) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_mgr)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as ac,
        app_with_mgr.router.lifespan_context(app_with_mgr),
    ):
        yield ac


# -----------------------------------------------------------------------------
# 1. CORS Origin and Credential Leakage Protection Tests
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cors_allowed_development_origin(client: AsyncClient) -> None:
    """Explicit development origins receive Access-Control-Allow-Origin header without wildcard or credentials."""
    resp = await client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    # Never allow credentials with open/broad CORS in production or local development
    assert resp.headers.get("access-control-allow-credentials") is None


@pytest.mark.anyio
async def test_cors_disallowed_origin(client: AsyncClient) -> None:
    """Unauthorized external origins are rejected and do not receive Access-Control-Allow-Origin."""
    resp = await client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://malicious-external-site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") is None


# -----------------------------------------------------------------------------
# 2. Filesystem Path Sanitization Tests
# -----------------------------------------------------------------------------


def test_path_sanitization_masks_windows_and_unix_paths() -> None:
    """Local filesystem paths in exception messages must be sanitized before being returned to clients."""
    windows_msg = "Failed to open SQLite database at C:\\Workspace\\Projects\\test.db: lock timeout"
    sanitized_win = sanitize_storage_error_message(windows_msg)
    assert "C:\\Workspace\\Projects\\test.db" not in sanitized_win
    assert sanitized_win == "A persistence layer storage error occurred."

    unix_msg = "Database file /home/runner/work/engine/data.db is corrupted"
    sanitized_unix = sanitize_storage_error_message(unix_msg)
    assert "/home/runner/work/engine/data.db" not in sanitized_unix
    assert sanitized_unix == "A persistence layer storage error occurred."

    mac_msg = "Could not create directory at /Users/developer/.cache/model.bin"
    sanitized_mac = sanitize_storage_error_message(mac_msg)
    assert "/Users/developer/.cache/model.bin" not in sanitized_mac
    assert sanitized_mac == "A persistence layer storage error occurred."

    safe_msg = "Entity not found in catalog"
    assert sanitize_storage_error_message(safe_msg) == safe_msg


# -----------------------------------------------------------------------------
# 3. Authoritative GPU Reference Catalog Endpoint Tests
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_hardware_catalog_specifications(client: AsyncClient) -> None:
    """GET /api/v1/hardware/catalog returns authoritative theoretical GPU specifications from catalog.py."""
    resp = await client.get("/api/v1/hardware/catalog")
    assert resp.status_code == 200
    catalog = resp.json()

    expected_specs = list_gpu_specs()
    assert isinstance(catalog, list)
    assert len(catalog) == len(expected_specs)

    # Verify H100 specification integrity
    h100_entries = [g for g in catalog if g["catalog_id"] == "nvidia-h100-sxm5-80gb"]
    assert len(h100_entries) == 1
    h100 = h100_entries[0]

    assert h100["name"] == "NVIDIA H100 SXM5 80GB"
    assert h100["architecture"] == "Hopper"
    assert h100["memory_capacity_gb"] == 80.0
    assert h100["theoretical_memory_bandwidth_gbps"] == 3350.0
    assert h100["compute"]["fp16_dense_tflops"] == 989.0
    assert h100["provenance"] == "HEURISTIC"
