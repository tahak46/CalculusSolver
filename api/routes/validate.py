"""
/validate route — no FastAPI, no pydantic.
Accepts: POST {"expression": {...SLaNg object...}}
Returns: validation result from validate_slang.js
"""

import json
import subprocess
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

_SCRIPT = Path(__file__).resolve().parents[1] / "validate_slang.js"


async def validate_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

    expression = body.get("expression", body)

    if not _SCRIPT.exists():
        return JSONResponse({"detail": "validate_slang.js not found."}, status_code=503)

    try:
        proc = subprocess.run(
            ["node", "--input-type=module", str(_SCRIPT)],
            input=json.dumps({"expression": expression}),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return JSONResponse(
            {"detail": "Node.js not found. Install Node.js to use validation."},
            status_code=503,
        )

    if proc.returncode != 0 and not proc.stdout.strip():
        return JSONResponse(
            {"detail": proc.stderr.strip() or "Validation failed."},
            status_code=500,
        )

    try:
        return JSONResponse(json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        return JSONResponse(
            {"detail": f"Validator parse error: {exc}"}, status_code=500
        )
