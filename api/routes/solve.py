"""
/solve route — no FastAPI, no pydantic.
Accepts: POST {"input": {...SLaNg envelope...}}
Returns: {"status", "expr", "steps", "latex", "confidence", "verified", "warning", "rule"}
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

_SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _format_latex(expression: Any) -> Optional[Any]:
    script = _SCRIPT_DIR / "format_slang_expression.js"
    if not script.exists():
        return None
    try:
        proc = subprocess.run(
            ["node", "--input-type=module", str(script)],
            input=json.dumps({"expression": expression}),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout).get("latex")
    except Exception:
        return None


def _unwrap_output(output: Any) -> dict:
    if isinstance(output, dict) and "expr" in output:
        return {"expr": output["expr"], "steps": output.get("steps", [])}
    return {"expr": output, "steps": []}


async def solve_route(request: Request, state: dict) -> JSONResponse:
    solver = state.get("solver")
    if solver is None:
        err = state.get("solver_error", "Solver not available.")
        return JSONResponse({"detail": err}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

    input_env = body.get("input")
    if not isinstance(input_env, dict):
        return JSONResponse(
            {"detail": "'input' must be a JSON object."}, status_code=422
        )

    try:
        result = solver.solve(input_env)
    except Exception as exc:
        return JSONResponse({"detail": f"Solver error: {exc}"}, status_code=500)

    unpacked = _unwrap_output(result.get("output"))
    latex = _format_latex(unpacked["expr"])

    return JSONResponse(
        {
            "status": result.get("status", "unverified"),
            "expr": unpacked["expr"],
            "steps": unpacked["steps"],
            "latex": latex,
            "confidence": float(result.get("confidence", 0.0)),
            "verified": result.get("verified"),
            "warning": result.get("warning"),
            "rule": result.get("rule"),
        }
    )
