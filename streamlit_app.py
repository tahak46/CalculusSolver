import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent


EXAMPLES = {
    "d/dx x^2": {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {"terms": [{"coeff": 1, "var": {"x": 2}}]},
            "deno": 1,
        },
    },
    "d/dx 3x^3 + 2x": {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 3, "var": {"x": 3}},
                    {"coeff": 2, "var": {"x": 1}},
                ]
            },
            "deno": 1,
        },
    },
    "integral 6x dx": {
        "op": "integrate",
        "var": "x",
        "expr": {
            "numi": {"terms": [{"coeff": 6, "var": {"x": 1}}]},
            "deno": 1,
        },
    },
}


def resolve_model_path() -> Path:
    candidates = [
        ROOT / "checkpoints" / "final" / "best.pt",
        ROOT / "checkpoints" / "sft" / "best.pt",
        ROOT / "checkpoints" / "pretrain" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No model checkpoint found. Tried: {tried}")


def _copy_fraction(expr):
    return json.loads(json.dumps(expr))


def _normalize_term(term):
    clean = {"coeff": term.get("coeff", 0)}
    variables = {key: value for key, value in term.get("var", {}).items() if value != 0}
    if variables:
        clean["var"] = variables
    return clean


def _differentiate_fraction(expr, variable):
    if expr.get("deno", 1) != 1:
        raise ValueError("Fallback solver only supports simple denominator 1.")

    terms = []
    for term in expr.get("numi", {}).get("terms", []):
        power = term.get("var", {}).get(variable, 0)
        if power == 0:
            continue
        next_term = _copy_fraction(term)
        next_term["coeff"] = next_term.get("coeff", 0) * power
        next_term.setdefault("var", {})[variable] = power - 1
        terms.append(_normalize_term(next_term))

    return {"numi": {"terms": terms or [{"coeff": 0}]}, "deno": 1}


def _integrate_fraction(expr, variable):
    if expr.get("deno", 1) != 1:
        raise ValueError("Fallback solver only supports simple denominator 1.")

    terms = []
    for term in expr.get("numi", {}).get("terms", []):
        power = term.get("var", {}).get(variable, 0)
        if power == -1:
            raise ValueError("Fallback solver does not support logarithmic integrals.")
        next_term = _copy_fraction(term)
        next_power = power + 1
        next_term["coeff"] = next_term.get("coeff", 0) / next_power
        next_term.setdefault("var", {})[variable] = next_power
        terms.append(_normalize_term(next_term))

    return {"numi": {"terms": terms or [{"coeff": 0}]}, "deno": 1}


def _term_to_text(term):
    coeff = term.get("coeff", 0)
    variables = term.get("var", {})
    if not variables:
        return str(coeff)

    pieces = []
    if coeff == -1:
        pieces.append("-")
    elif coeff != 1:
        pieces.append(str(coeff))

    for name, power in variables.items():
        pieces.append(name if power == 1 else f"{name}^{power}")
    return "".join(pieces)


def _fraction_to_text(expr):
    terms = expr.get("numi", {}).get("terms", [])
    numerator = " + ".join(_term_to_text(term) for term in terms).replace("+ -", "- ")
    denominator = expr.get("deno", 1)
    if denominator == 1:
        return numerator
    return f"({numerator}) / ({denominator})"


class FallbackSolver:
    mode = "fallback"

    def solve(self, payload):
        operation = payload.get("op")
        variable = payload.get("var", "x")
        expr = payload.get("expr")
        if not isinstance(expr, dict):
            raise ValueError("Input must include expr as a SLaNg fraction object.")

        if operation in ("diff", "partial"):
            output = _differentiate_fraction(expr, variable)
            rule = "power_rule"
        elif operation == "integrate":
            output = _integrate_fraction(expr, variable)
            rule = "power_rule_integral"
        else:
            raise ValueError(
                "Fallback solver supports diff, partial, and integrate. "
                "Add a trained checkpoint for full model inference."
            )

        return {
            "status": "solved",
            "expr": output,
            "steps": [
                {
                    "rule": rule,
                    "description": "Solved with deterministic fallback calculus.",
                }
            ],
            "latex": _fraction_to_text(output),
            "confidence": 1.0,
            "verified": True,
            "warning": "Fallback mode: no neural checkpoint is loaded.",
        }


@st.cache_resource
def load_solver():
    try:
        model_path = resolve_model_path()
        from inference.solve import CalculusSolverInference

        solver = CalculusSolverInference(
            model_path=str(model_path),
            vocab_path=str(ROOT / "tokenizer" / "vocab.json"),
            beam_size=5,
            max_len=256,
        )
        solver.mode = "neural"
        return solver, None
    except Exception as exc:
        return FallbackSolver(), str(exc)


st.set_page_config(page_title="CalculusSolver", layout="wide")
st.title("CalculusSolver")

solver, solver_error = load_solver()
if solver_error:
    st.info(
        "No trained checkpoint is deployed yet, so the app is running in fallback "
        "mode for basic polynomial differentiation and integration."
    )
    with st.expander("Checkpoint details"):
        st.code(solver_error)
else:
    st.success("Neural checkpoint loaded.")

left, right = st.columns([0.45, 0.55])

with left:
    selected = st.selectbox("Example", list(EXAMPLES.keys()))
    default_payload = json.dumps(EXAMPLES[selected], indent=2)
    raw_input = st.text_area("Input envelope", value=default_payload, height=320)
    run = st.button("Solve", type="primary", use_container_width=True)

with right:
    st.subheader("Result")
    if run:
        try:
            payload = json.loads(raw_input)
            result = solver.solve(payload)
            st.json(result)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.caption("Choose an example or edit the JSON, then press Solve.")
