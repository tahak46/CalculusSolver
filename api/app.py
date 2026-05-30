"""
CalculusSolver API — Starlette only, no FastAPI, no pydantic.
Compatible with Python 3.10+ including 3.14.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route, Mount

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.solve import solve_route
from api.routes.validate import validate_route

# ── Shared app state (plain dict, no pydantic) ────────────────────────────────
_state: dict = {"solver": None, "solver_error": None}


def _resolve_model_path() -> str:
    candidates = []
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        candidates.append(p if p.is_absolute() else ROOT / p)
    candidates += [
        ROOT / "checkpoints" / "final" / "best.pt",
        ROOT / "checkpoints" / "sft" / "best.pt",
        ROOT / "checkpoints" / "pretrain" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No checkpoint found. Tried: " + ", ".join(str(p) for p in candidates)
    )


async def _startup():
    try:
        from inference.solve import CalculusSolverInference

        model_path = _resolve_model_path()
        _state["solver"] = CalculusSolverInference(
            model_path=model_path,
            vocab_path=str(ROOT / "tokenizer" / "vocab.json"),
            beam_size=5,
            max_len=256,
        )
        print(f"CalculusSolver loaded: {model_path}", flush=True)
    except Exception as exc:
        _state["solver"] = None
        _state["solver_error"] = str(exc)
        print(f"CalculusSolver started WITHOUT model: {exc}", flush=True)


async def _shutdown():
    solver = _state.get("solver")
    if solver is not None:
        try:
            solver.close()
        except Exception:
            pass


from starlette.responses import JSONResponse


async def health(request):
    return JSONResponse(
        {
            "status": "ok",
            "solver_loaded": _state["solver"] is not None,
            "solver_error": _state["solver_error"],
        }
    )


ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000",
).split(",")

app = Starlette(
    debug=False,
    routes=[
        Route("/health", health),
        Route("/solve", lambda req: solve_route(req, _state)),
        Route("/validate", validate_route),
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
    on_startup=[_startup],
    on_shutdown=[_shutdown],
)
