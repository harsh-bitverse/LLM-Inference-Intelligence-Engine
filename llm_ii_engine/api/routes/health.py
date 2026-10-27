"""Health check endpoint evaluating service and database liveness and readiness."""

from fastapi import APIRouter, Depends, Response, status

from llm_ii_engine.api.dependencies import get_db_manager
from llm_ii_engine.storage.database import DatabaseManager

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Service and database health probe")
async def health_check(
    response: Response,
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> dict[str, str]:
    """Return health status, verifying database responsiveness and connection pool liveness."""
    is_healthy = db_manager.check_health()
    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "database": "disconnected",
        }

    return {
        "status": "ok",
        "database": "connected",
        "version": "1.0.0",
    }
