__all__ = ["CalculusSolverInference"]


def __getattr__(name):
    if name == "CalculusSolverInference":
        from .solve import CalculusSolverInference

        return CalculusSolverInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
