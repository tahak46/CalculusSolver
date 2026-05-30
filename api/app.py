"""
CalculusSolver API — Starlette only, no FastAPI, no pydantic.
Compatible with Python 3.10+ including 3.14.

Startup behaviour:
  1. Try to load a trained neural checkpoint (checkpoints/final → sft → pretrain).
  2. If none found, automatically fall back to FallbackSolver (pure-Python
     polynomial solver — no torch, no checkpoint required).
  The API always starts successfully; /health reports which mode is active.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.solve import solve_route
from api.routes.validate import validate_route

# ── Shared state ──────────────────────────────────────────────────────────────
_state: dict = {
    "solver": None,
    "solver_mode": "unloaded",  # "neural" | "fallback" | "unloaded"
    "solver_stage": None,  # "final" | "sft" | "pretrain" | None
    "solver_error": None,
}

CHECKPOINT_PRIORITY = [
    ("final", ROOT / "checkpoints" / "final" / "best.pt"),
    ("sft", ROOT / "checkpoints" / "sft" / "best.pt"),
    ("pretrain", ROOT / "checkpoints" / "pretrain" / "best.pt"),
]


def _resolve_model_path():
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        p = p if p.is_absolute() else ROOT / p
        if p.exists():
            return str(p), "env"
    for stage, path in CHECKPOINT_PRIORITY:
        if path.exists():
            return str(path), stage
    return None, None


async def _startup():
    model_path, stage = _resolve_model_path()

    if model_path:
        # ── Try neural model ──────────────────────────────────────────────────
        try:
            from inference.solve import CalculusSolverInference

            solver = CalculusSolverInference(
                model_path=model_path,
                vocab_path=str(ROOT / "tokenizer" / "vocab.json"),
                beam_size=5,
                max_len=256,
            )
            _state["solver"] = solver
            _state["solver_mode"] = "neural"
            _state["solver_stage"] = stage
            _state["solver_error"] = None
            print(
                f"[CalculusSolver] Neural model loaded — stage={stage}, path={model_path}",
                flush=True,
            )
            return
        except Exception as exc:
            _state["solver_error"] = str(exc)
            print(f"[CalculusSolver] Neural load failed: {exc}", flush=True)

    # ── Fall back to deterministic solver ─────────────────────────────────────
    from inference.fallback_solver import FallbackSolver

    _state["solver"] = FallbackSolver()
    _state["solver_mode"] = "fallback"
    _state["solver_stage"] = None
    if not _state["solver_error"]:
        _state["solver_error"] = "No checkpoint found. Tried: " + ", ".join(
            str(p) for _, p in CHECKPOINT_PRIORITY
        )
    print(
        "[CalculusSolver] Running in FALLBACK mode — "
        "supports diff, partial, integrate, gradient, tangent_line. "
        "Train a checkpoint to enable full neural inference.",
        flush=True,
    )


async def _shutdown():
    solver = _state.get("solver")
    if solver is not None and hasattr(solver, "close"):
        try:
            solver.close()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app):
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


# ── Routes ────────────────────────────────────────────────────────────────────


async def health(request: Request):
    return JSONResponse(
        {
            "status": "ok",
            "solver_mode": _state["solver_mode"],
            "solver_stage": _state["solver_stage"],
            "solver_loaded": _state["solver"] is not None,
            "checkpoint_error": (
                _state["solver_error"] if _state["solver_mode"] != "neural" else None
            ),
        }
    )


ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://127.0.0.1:8002",
).split(",")

app = Starlette(
    debug=False,
    routes=[
        Route("/health", health),
        Route("/solve", lambda req: solve_route(req, _state), methods=["POST"]),
        Route("/validate", validate_route, methods=["POST"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
    lifespan=lifespan,
)
