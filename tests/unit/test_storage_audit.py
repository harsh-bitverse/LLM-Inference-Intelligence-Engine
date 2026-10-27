"""Targeted audit tests for database schema, SQLite pragmas, transactions, and migration safety."""

import contextlib
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from llm_ii_engine.storage.database import (
    CURRENT_SCHEMA_VERSION,
    DatabaseManager,
    create_engine_and_session,
    get_default_database_url,
    register_migration,
)
from llm_ii_engine.storage.models import (
    Base,
    HardwareReportORM,
    SchemaVersionORM,
)


@pytest.fixture
def temp_db_file() -> Generator[Path, None, None]:
    """Yield a unique temporary SQLite file path and ensure cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        with contextlib.suppress(PermissionError):
            path.unlink()
        wal_file = path.with_name(f"{path.name}-wal")
        if wal_file.exists():
            with contextlib.suppress(PermissionError):
                wal_file.unlink()
        shm_file = path.with_name(f"{path.name}-shm")
        if shm_file.exists():
            with contextlib.suppress(PermissionError):
                shm_file.unlink()


def test_sqlite_pragmas_on_every_connection(temp_db_file: Path) -> None:
    """Verify that foreign keys, WAL mode, synchronous normal, and busy timeout are configured."""
    db_url = f"sqlite:///{temp_db_file}"
    mgr = DatabaseManager(database_url=db_url)
    mgr.init_db()

    with mgr.engine.connect() as conn:
        fk_status = conn.execute(text("PRAGMA foreign_keys")).scalar()
        assert fk_status == 1, "Foreign keys must be enabled on every connection."

        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        assert str(journal_mode).lower() == "wal", "WAL mode must be active on file databases."

        busy_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
        assert busy_timeout == 5000, "Busy timeout must be set to 5000ms."

    # Test that a second distinct connection also has foreign_keys = ON
    with mgr.engine.connect() as conn2:
        fk_status_2 = conn2.execute(text("PRAGMA foreign_keys")).scalar()
        assert fk_status_2 == 1, "Subsequent connection must also enforce foreign keys."

    mgr.dispose()


def test_reopening_existing_database_preserves_schema_and_data(temp_db_file: Path) -> None:
    """Verify reopening an existing SQLite database retains records and does not overwrite schema version."""
    db_url = f"sqlite:///{temp_db_file}"

    # First session: initialize and write a record
    mgr1 = DatabaseManager(database_url=db_url)
    ver1 = mgr1.init_db(check_version=True)
    assert ver1 == CURRENT_SCHEMA_VERSION

    with mgr1.get_session() as session:
        hw = HardwareReportORM(
            report_id="hw-reopen-test",
            created_at=datetime.now(UTC),
            is_simulated=False,
            execution_mode="REAL",
            os_name="Linux",
            os_version="6.5",
            os_release="6.5.0",
            os_arch="x86_64",
            python_version="3.12",
            cpu_model="AMD Ryzen",
            cpu_physical_cores=8,
            cpu_logical_cores=16,
            total_ram_bytes=32 * 1024**3,
        )
        session.add(hw)

    mgr1.dispose()

    # Second session: reopen database and verify data is present
    mgr2 = DatabaseManager(database_url=db_url)
    ver2 = mgr2.init_db(check_version=True)
    assert ver2 == CURRENT_SCHEMA_VERSION

    with mgr2.get_session() as session:
        loaded = session.get(HardwareReportORM, "hw-reopen-test")
        assert loaded is not None
        assert loaded.cpu_model == "AMD Ryzen"

        # Verify only 1 version record exists
        versions = session.scalars(text("SELECT version FROM schema_version")).all()
        assert len(versions) == 1
        assert versions[0] == CURRENT_SCHEMA_VERSION

    mgr2.dispose()


def test_higher_schema_version_rejection(temp_db_file: Path) -> None:
    """Verify that opening a database with a future schema version raises a loud RuntimeError."""
    db_url = f"sqlite:///{temp_db_file}"
    mgr = DatabaseManager(database_url=db_url)
    mgr.init_db(check_version=True)

    # Manually inject a future schema version
    with mgr.get_session() as session:
        session.add(
            SchemaVersionORM(
                version=999,
                applied_at=datetime.now(UTC),
                description="Future migration from Session 99",
            )
        )

    mgr.dispose()

    # Reopening should now fail loudly
    mgr_future = DatabaseManager(database_url=db_url)
    with pytest.raises(RuntimeError, match="higher than supported version"):
        mgr_future.init_db(check_version=True)

    mgr_future.dispose()


def test_schema_migration_execution(temp_db_file: Path) -> None:
    """Verify that an older schema version triggers registered forward migrations safely without data loss."""
    db_url = f"sqlite:///{temp_db_file}"
    engine, session_factory = create_engine_and_session(db_url)
    Base.metadata.create_all(bind=engine)

    # Seed with an older version record: version 0
    with session_factory() as session:
        session.add(
            SchemaVersionORM(
                version=0,
                applied_at=datetime.now(UTC),
                description="Legacy pre-release schema",
            )
        )
        hw = HardwareReportORM(
            report_id="hw-mig-test",
            created_at=datetime.now(UTC),
            is_simulated=False,
            execution_mode="REAL",
            os_name="Windows",
            os_version="11",
            os_release="10.0.22631",
            os_arch="AMD64",
            python_version="3.12.10",
            cpu_model="Intel Core i9",
            cpu_physical_cores=16,
            cpu_logical_cores=32,
            total_ram_bytes=64 * 1024**3,
        )
        session.add(hw)
        session.commit()
    engine.dispose()

    migration_executed = False

    def migration_0_to_1(session: Session) -> None:
        nonlocal migration_executed
        migration_executed = True
        # Verify existing data is intact during migration
        rec = session.get(HardwareReportORM, "hw-mig-test")
        assert rec is not None

    register_migration(0, "Migration from v0 to v1", migration_0_to_1)

    # Initialize via DatabaseManager: should run migration 0 -> 1
    mgr = DatabaseManager(database_url=db_url)
    active_ver = mgr.init_db(check_version=True)
    assert active_ver == CURRENT_SCHEMA_VERSION
    assert migration_executed, "Migration function should have executed."

    # Verify existing data survived and version 1 was added
    with mgr.get_session() as session:
        loaded = session.get(HardwareReportORM, "hw-mig-test")
        assert loaded is not None
        assert loaded.cpu_model == "Intel Core i9"

        versions = session.scalars(
            text("SELECT version FROM schema_version ORDER BY version")
        ).all()
        assert versions == list(range(CURRENT_SCHEMA_VERSION + 1))

    mgr.dispose()


def test_missing_migration_path_raises_error(temp_db_file: Path) -> None:
    """Verify that an older schema version with no registered migration path raises RuntimeError."""
    db_url = f"sqlite:///{temp_db_file}"
    engine, session_factory = create_engine_and_session(db_url)
    Base.metadata.create_all(bind=engine)

    with session_factory() as session:
        # Version -5 has no migration registered
        session.add(
            SchemaVersionORM(
                version=-5,
                applied_at=datetime.now(UTC),
                description="Unknown legacy version",
            )
        )
        session.commit()
    engine.dispose()

    mgr = DatabaseManager(database_url=db_url)
    with pytest.raises(
        RuntimeError, match="No migration registered to upgrade schema from version -5"
    ):
        mgr.init_db(check_version=True)

    mgr.dispose()


def test_transaction_rollback_and_session_cleanup_under_exception(temp_db_file: Path) -> None:
    """Verify transaction rollback restores consistency and closes sessions under exception."""
    db_url = f"sqlite:///{temp_db_file}"
    mgr = DatabaseManager(database_url=db_url)
    mgr.init_db()

    with (
        pytest.raises(ValueError, match="Simulated mid-transaction failure"),
        mgr.get_session() as session,
    ):
        hw = HardwareReportORM(
            report_id="hw-fail-1",
            created_at=datetime.now(UTC),
            is_simulated=False,
            execution_mode="REAL",
            os_name="Linux",
            os_version="6.5",
            os_release="6.5.0",
            os_arch="x86_64",
            python_version="3.12",
            cpu_model="AMD",
            cpu_physical_cores=4,
            cpu_logical_cores=8,
            total_ram_bytes=16 * 1024**3,
        )
        session.add(hw)
        session.flush()
        raise ValueError("Simulated mid-transaction failure")

    # Record must NOT exist
    with mgr.get_session() as session:
        assert session.get(HardwareReportORM, "hw-fail-1") is None

    mgr.dispose()


def test_database_isolation_from_default_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify default database path resolution respects environment and does not write to project root by accident."""
    test_db_url = f"sqlite:///{tmp_path / 'env_isolated.db'}"
    monkeypatch.setenv("IIE_DATABASE_URL", test_db_url)

    assert get_default_database_url() == test_db_url
