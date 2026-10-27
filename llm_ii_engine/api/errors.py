"""Standardized error responses and HTTP exception handlers for FastAPI."""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_ii_engine.storage.repository import (
    EntityAlreadyExistsError,
    EntityNotFoundError,
    StorageError,
)


class ErrorDetail(BaseModel):
    """Standardized machine-readable and human-readable error payload."""

    error: str = Field(description="Error code or category")
    message: str = Field(description="Human-readable explanation of error")
    detail: Any | None = Field(default=None, description="Detailed context or field errors")


def sanitize_storage_error_message(err_msg: str) -> str:
    """Prevent leaking absolute filesystem paths, database URIs, or internal driver errors."""
    if any(token in err_msg for token in (":\\", ":/", "/home/", "/Users/", "/var/", "sqlite://")):
        return "A persistence layer storage error occurred."
    return err_msg


def register_error_handlers(app: FastAPI) -> None:
    """Register uniform exception handlers mapping domain and storage errors to HTTP codes."""

    @app.exception_handler(EntityNotFoundError)
    async def not_found_handler(request: Request, exc: EntityNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorDetail(
                error="ENTITY_NOT_FOUND",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(EntityAlreadyExistsError)
    async def conflict_handler(request: Request, exc: EntityAlreadyExistsError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=ErrorDetail(
                error="ENTITY_ALREADY_EXISTS",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorDetail(
                error="BAD_REQUEST",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorDetail(
                error="VALIDATION_ERROR",
                message="Request payload failed schema validation",
                detail=exc.errors(),
            ).model_dump(),
        )

    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError) -> JSONResponse:
        sanitized_msg = sanitize_storage_error_message(str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorDetail(
                error="STORAGE_ERROR",
                message=sanitized_msg,
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDetail(
                error="HTTP_ERROR",
                message=exc.detail if isinstance(exc.detail, str) else "HTTP error",
                detail=exc.detail if not isinstance(exc.detail, str) else None,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorDetail(
                error="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred.",
            ).model_dump(),
        )
