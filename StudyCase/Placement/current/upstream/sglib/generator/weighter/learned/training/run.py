from .cli import main
from .engine import run_training_task

__all__ = ["main", "run_training_task"]

if __name__ == "__main__":
    raise SystemExit(main())

