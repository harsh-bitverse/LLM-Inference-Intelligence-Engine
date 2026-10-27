"""Database engine configuration, connection lifecycle, and schema initialization.

Configures SQLite connection settings (foreign keys, WAL mode, timeouts)
and schema initialization with version tracking.
"""

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.orm import Session, sessionmaker

from llm_ii_engine.storage.models import Base, SchemaVersionORM

CURRENT_SCHEMA_VERSION = 2
DEFAULT_DB_PATH = "storage/llm_ii_engine.db"

MigrationFn = Callable[[Session], None]
SCHEMA_MIGRATIONS: dict[int, tuple[str, MigrationFn]] = {}


def register_migration(from_version: int, description: str, fn: MigrationFn) -> None:
    """Register a forward schema migration function from from_version to from_version + 1."""
    SCHEMA_MIGRATIONS[from_version] = (description, fn)


def _column_exists(session: Session, table: str, column: str) -> bool:
    """Check if a column already exists in a SQLite table."""
    result = session.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return any(row[1] == column for row in result)


def _migration_1_to_2(session: Session) -> None:
    """Migration 1 -> 2: Add ttfc_ms to request_benchmarks and chunk aggregates to benchmark_suites."""
    if not _column_exists(session, "request_benchmarks", "ttfc_ms"):
        session.execute(text("ALTER TABLE request_benchmarks ADD COLUMN ttfc_ms FLOAT"))
    if not _column_exists(session, "benchmark_suites", "ttfc_p50_ms"):
        session.execute(text("ALTER TABLE benchmark_suites ADD COLUMN ttfc_p50_ms FLOAT"))
    if not _column_exists(session, "benchmark_suites", "ttfc_p90_ms"):
        session.execute(text("ALTER TABLE benchmark_suites ADD COLUMN ttfc_p90_ms FLOAT"))
    if not _column_exists(session, "benchmark_suites", "ttfc_p95_ms"):
        session.execute(text("ALTER TABLE benchmark_suites ADD COLUMN ttfc_p95_ms FLOAT"))
    if not _column_exists(session, "benchmark_suites", "ttfc_p99_ms"):
        session.execute(text("ALTER TABLE benchmark_suites ADD COLUMN ttfc_p99_ms FLOAT"))
    if not _column_exists(session, "benchmark_suites", "chunk_latency_p50_ms"):
        session.execute(text("ALTER TABLE benchmark_suites ADD COLUMN chunk_latency_p50_ms FLOAT"))


register_migration(
    1,
    "Add ttfc_ms to request_benchmarks and chunk latency aggregates to benchmark_suites",
    _migration_1_to_2,
)


def get_default_database_url() -> str:
    """Resolve the default SQLite database URL from environment or project directory."""
    url = os.environ.get("IIE_DATABASE_URL")
    if url:
        return url

    # Default to storage directory relative to project root / current working dir
    storage_dir = Path("storage")
    storage_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_dir / 'llm_ii_engine.db'}"


def configure_sqlite_pragmas(engine: Engine) -> None:
    """Attach connect listeners to enforce SQLite foreign keys, WAL mode, and busy timeouts."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            # Enforce foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            # Enable write-ahead logging for high concurrency (readers do not block writers)
            cursor.execute("PRAGMA journal_mode = WAL")
            # Normal sync mode in WAL gives durability while avoiding fsync stalls
            cursor.execute("PRAGMA synchronous = NORMAL")
            # Set 5000ms busy timeout to prevent immediate database locked errors
            cursor.execute("PRAGMA busy_timeout = 5000")
        finally:
            cursor.close()


def create_engine_and_session(
    database_url: str | None = None,
    echo: bool = False,
) -> tuple[Engine, sessionmaker[Session]]:
    """Create a configured SQLAlchemy Engine and sessionmaker.

    Args:
        database_url: SQLite database URL. Defaults to environment or local storage.
        echo: Whether to log SQL statements.

    Returns:
        Tuple of (Engine, sessionmaker).
    """
    url = database_url or get_default_database_url()

    # Ensure parent directories exist for file-based SQLite databases
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        db_file = url.removeprefix("sqlite:///")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}

    engine = create_engine(
        url,
        echo=echo,
        connect_args=connect_args,
    )

    if url.startswith("sqlite"):
        configure_sqlite_pragmas(engine)

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    return engine, session_factory


class DatabaseManager:
    """Manages the database connection lifecycle, sessions, and schema migrations."""

    def __init__(
        self,
        database_url: str | None = None,
        echo: bool = False,
    ) -> None:
        self._database_url = database_url or get_default_database_url()
        self._echo = echo
        self._closed: bool = False
        self._engine, self._session_factory = create_engine_and_session(
            database_url=self._database_url,
            echo=self._echo,
        )

    @property
    def engine(self) -> Engine:
        """Get the active SQLAlchemy Engine."""
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Get the configured sessionmaker."""
        return self._session_factory

    @property
    def database_url(self) -> str:
        """Get the configured database URL."""
        return self._database_url

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Yield a managed session within a transactional context."""
        if self._closed:
            raise RuntimeError("DatabaseManager has been disposed.")
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def check_health(self) -> bool:
        """Verify database connectivity and responsiveness."""
        if self._closed:
            return False
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                return result == 1
        except Exception:
            return False

    def init_db(self, check_version: bool = True) -> int:
        """Initialize all relational database tables and record schema version.

        Args:
            check_version: Whether to verify and update the schema_version table.

        Returns:
            The active schema version integer.

        Raises:
            RuntimeError: If existing database schema version exceeds application code version
                          or if an upgrade path is missing.
        """
        # Create all declared tables if they do not exist
        Base.metadata.create_all(bind=self._engine)

        if not check_version:
            return CURRENT_SCHEMA_VERSION

        with self._session_factory() as session:
            try:
                # Query highest recorded version
                stmt = select(SchemaVersionORM).order_by(SchemaVersionORM.version.desc()).limit(1)
                latest_ver = session.scalars(stmt).first()

                if latest_ver is None:
                    # Fresh database initialization: record current version
                    session.add(
                        SchemaVersionORM(
                            version=CURRENT_SCHEMA_VERSION,
                            applied_at=datetime.now(UTC),
                            description=f"Initial relational schema v{CURRENT_SCHEMA_VERSION}",
                        )
                    )
                    session.commit()
                    return CURRENT_SCHEMA_VERSION

                if latest_ver.version > CURRENT_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Database schema version {latest_ver.version} is higher than supported "
                        f"version {CURRENT_SCHEMA_VERSION}. Please upgrade application code."
                    )

                if latest_ver.version < CURRENT_SCHEMA_VERSION:
                    # Run registered migrations sequentially from latest_ver.version to CURRENT_SCHEMA_VERSION
                    for ver in range(latest_ver.version, CURRENT_SCHEMA_VERSION):
                        if ver not in SCHEMA_MIGRATIONS:
                            raise RuntimeError(
                                f"No migration registered to upgrade schema from version {ver} to {ver + 1}."
                            )
                        desc, migration_fn = SCHEMA_MIGRATIONS[ver]
                        migration_fn(session)
                        session.add(
                            SchemaVersionORM(
                                version=ver + 1,
                                applied_at=datetime.now(UTC),
                                description=desc,
                            )
                        )
                        session.flush()
                    session.commit()
                    return CURRENT_SCHEMA_VERSION

                return latest_ver.version
            except Exception:
                session.rollback()
                raise

    def dispose(self) -> None:
        """Dispose of the connection pool."""
        self._closed = True
        self._engine.dispose()
