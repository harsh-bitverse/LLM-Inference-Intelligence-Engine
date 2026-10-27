"""FastAPI dependency injection providers for database sessions, repository, and job manager."""

from collections.abc import Generator
from typing import Any, cast

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.repository import EngineRepository


def get_db_manager(request: Request) -> DatabaseManager:
    """Extract the application-wide DatabaseManager from app state."""
    return cast(DatabaseManager, request.app.state.db_manager)


def get_db_session(
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> Generator[Session, None, None]:
    """Yield a transactional database session."""
    with db_manager.get_session() as session:
        yield session


def get_repository(
    session: Session = Depends(get_db_session),
) -> EngineRepository:
    """Provide an EngineRepository bound to the current request's session."""
    return EngineRepository(session)


def get_job_manager(request: Request) -> Any:
    """Extract the BenchmarkJobManager from app state."""
    return request.app.state.job_manager
