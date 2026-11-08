"""FastAPI backend and static SvelteKit UI. Run with ONE uvicorn worker."""
from contextlib import asynccontextmanager
import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service import Console

ROOT = Path(__file__).resolve().parents[1]


def create_app(config=None, console=None):
    @asynccontextmanager
    async def lifespan(app):
        nonlocal console
        if console is None:
            file = os.environ.get("GR8_CONSOLE_CONFIG")
            actual = config if config is not None else json.loads(Path(file).read_text()) if file else {}
            console = Console(actual)
        app.state.console = console
        yield
        await run_in_threadpool(console.close)

    app = FastAPI(title="gr8 console", lifespan=lifespan, docs_url=None, redoc_url=None)

    hosts = (config or (console.config if console else {})).get("allowed_hosts", ["localhost"])
    if config is None and console is None and os.environ.get("GR8_CONSOLE_CONFIG"):
        hosts = json.loads(Path(os.environ["GR8_CONSOLE_CONFIG"]).read_text()).get("allowed_hosts", ["localhost"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    def ok(value, status=200):
        return JSONResponse({"ok": True, "value": value}, status_code=status, headers={"Cache-Control": "no-store"})

    @app.middleware("http")
    async def guard(request, next_call):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            expected = str(request.base_url).rstrip("/")
            # CSRF token is read same-origin from snapshot; there is no permissive CORS.
            if (origin and origin != expected) or request.headers.get("x-console-token") != app.state.console.csrf:
                return JSONResponse({"ok": False, "error": "same-origin console token required"}, status_code=403)
            if "application/json" not in request.headers.get("content-type", ""):
                return JSONResponse({"ok": False, "error": "JSON required"}, status_code=415)
        response = await next_call(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/api/healthz")
    def healthz():
        return ok({"status": "up"})

    @app.get("/api/snapshot")
    def snapshot():
        return ok(app.state.console.snapshot())

    @app.get("/api/presets/{model}")
    def get_presets(model):
        try:
            return ok(app.state.console.read_presets(model))
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.post("/api/operations")
    async def operate(request: Request):
        try:
            body = await request.body()
            if len(body) > 16384:
                raise ValueError("request too large")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("expected object")
            return ok(await run_in_threadpool(app.state.console.submit, data), 202)
        except (ValueError, PermissionError, RuntimeError) as exc:
            status = 403 if isinstance(exc, PermissionError) else 409 if isinstance(exc, RuntimeError) else 400
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)

    @app.get("/api/operations/current")
    def operation():
        return ok(app.state.console.operation())

    @app.get("/{path:path}")
    def spa(path: str):
        build = (ROOT / "web/build").resolve()
        target = (build / path).resolve()
        if path.startswith("api/"):
            return JSONResponse({"ok": False, "error": "unknown endpoint"}, status_code=404)
        if path and target.is_relative_to(build) and target.is_file():
            return FileResponse(target)
        if (build / "index.html").exists():
            return FileResponse(build / "index.html")
        return JSONResponse({"ok": False, "error": "Build tools/console/web first"}, status_code=503)

    return app


app = create_app()
