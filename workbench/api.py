"""Loopback REST adapter for local decisions; this is not a Jev protocol server."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import threading
from typing import Annotated, Any

# Inherited tracing settings must not export local task content.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_OTEL_ENABLED'] = 'false'

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .backends import DecisionBackend, DecisionTuneBackend, RuleBackend, checked_decision
from .workflow import run_workflow

BODY_LIMIT = 256 * 1024
STATE_LIMIT = 65536
Identifier = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128, pattern=r'\S')]
Question = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2048, pattern=r'\S')]
OptionKey = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64, pattern=r'\S')]
OptionDescription = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512, pattern=r'\S')]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({'error': {'code': code, 'message': message}}, status_code=status)


def serialized(value: Any) -> bytes:
    """Compact UTF-8 JSON, rejecting nonfinite numbers and surrogate code points."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Repeated JSON object key.')
        result[key] = value
    return result


class BoundedJsonBody:
    """Bound received bytes before the framework attempts JSON parsing."""
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or scope['method'] != 'POST':
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            size += len(chunk)
            if size > BODY_LIMIT:
                await error_response(413, 'payload_too_large', 'Request body exceeds 256 KiB.')(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get('more_body', False):
                break
        body = b''.join(chunks)
        media_types = [value.decode('latin1').split(';', 1)[0].strip().lower()
                       for key, value in scope['headers'] if key.lower() == b'content-type']
        if media_types != ['application/json']:
            await error_response(400, 'invalid_request', 'Content-Type must be application/json.')(scope, receive, send)
            return
        try:
            parsed = json.loads(body.decode('utf-8'), object_pairs_hook=unique_object)
            serialized(parsed)
        except (ValueError, UnicodeError, RecursionError):
            await error_response(400, 'invalid_request', 'Request must contain valid UTF-8 JSON.')(scope, receive, send)
            return

        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {'type': 'http.request', 'body': body, 'more_body': False}

        await self.app(scope, replay, send)


class StrictBody(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class DecisionRequest(StrictBody):
    state: dict[str, JsonValue] | list[JsonValue] | str
    question: Question
    options: Annotated[dict[OptionKey, OptionDescription], Field(min_length=2, max_length=20)]


class Document(StrictBody):
    id: Identifier
    text: str


class Invoice(StrictBody):
    id: Identifier
    currency: str
    amount: str


class LedgerEntry(StrictBody):
    id: Identifier
    tenantId: Identifier
    invoiceId: Identifier
    currency: str
    amount: str


class ReviewRequest(StrictBody):
    id: Identifier
    tenantId: Identifier
    document: Document | None = None
    invoice: Invoice | None = None
    ledger: Annotated[list[LedgerEntry], Field(max_length=1000)] | None = None


def create_app(backend: DecisionBackend | None = None) -> FastAPI:
    """Use an explicitly supplied local provider, serialized across both routes."""
    provider = backend if backend is not None else RuleBackend()
    provider_lock = threading.Lock()
    app = FastAPI(title='Decision Workbench local API', version='0.1.0',
                  docs_url=None, redoc_url=None)
    app.add_middleware(BoundedJsonBody)

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, error: ApiError) -> JSONResponse:
        return error_response(error.status, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return error_response(400, 'invalid_request', 'Request validation failed.')

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        return error_response(error.status_code, 'not_found' if error.status_code == 404 else 'invalid_request',
                              'Route was not found.' if error.status_code == 404 else 'Request is not supported.')

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _error: Exception) -> JSONResponse:
        return error_response(500, 'internal_error', 'The request could not be completed.')

    @app.get('/healthz')
    def health() -> dict:
        return {'status': 'ok', 'backend': provider.name}

    @app.post('/v1/decisions')
    def decide(body: DecisionRequest) -> dict:
        if len(serialized(body.state)) > STATE_LIMIT:
            raise ApiError(413, 'payload_too_large', 'State exceeds 65536 serialized UTF-8 bytes.')
        try:
            with provider_lock, tracing_context(enabled=False):
                raw = provider.choose(deepcopy(body.state), body.question, dict(body.options))
                # Validate again at the HTTP boundary even for a custom backend.
                if not isinstance(raw.backend, str) or not raw.backend.strip():
                    raise ValueError('Invalid backend name.')
                result = checked_decision(raw.choice, dict(raw.probabilities), body.options, raw.backend)
                payload = result.as_dict()
                serialized(payload)
                return payload
        except Exception:
            raise ApiError(503, 'backend_unavailable', 'The local decision backend could not complete this request.') from None

    @app.post('/v1/reviews')
    def review(body: ReviewRequest) -> dict:
        task = body.model_dump(exclude_unset=True)
        try:
            with provider_lock:
                # Workflow gates and raw decision traces stay visible in a 200
                # result; the adapter never converts a review into an action.
                result = run_workflow(task, provider)
                serialized(result)
                return result
        except Exception:
            raise ApiError(503, 'backend_unavailable', 'The local review backend could not complete this request.') from None

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a loopback typed-decision study API.')
    parser.add_argument('--backend', choices=('rules', 'decisiontune'), default='rules')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--engine', choices=('mlx',), default='mlx')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535.')
    if args.backend == 'decisiontune' and args.model_path is None:
        parser.error('--model-path is required for the decisiontune backend.')
    try:
        provider = (RuleBackend() if args.backend == 'rules' else
                    DecisionTuneBackend(args.model_path, backend=args.engine))
    except Exception:
        parser.exit(1, 'Local backend could not load; check model files and runtime.\n')

    import uvicorn

    # Bind once so port 0 can announce its actual address without a port race.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', args.port))
        listener.listen(128)
        print(json.dumps({'address': f'http://127.0.0.1:{listener.getsockname()[1]}',
                          'backend': provider.name}), flush=True)
        config = uvicorn.Config(create_app(provider), log_level='warning', access_log=False)
        uvicorn.Server(config).run(sockets=[listener])


if __name__ == '__main__':
    main()
