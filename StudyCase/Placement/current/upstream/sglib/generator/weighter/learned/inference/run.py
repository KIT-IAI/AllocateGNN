from .cli import main
from .engine import run_inference_task

__all__ = ["main", "run_inference_task"]

if __name__ == "__main__":
    raise SystemExit(main())

