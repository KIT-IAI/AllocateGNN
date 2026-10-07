"""Experiment 的只读表交接。"""

from pathlib import Path
import json
import pandas as pd

from sglib.core.infra.content_chain import verify_chain
from sglib.core.infra.hashing import sha256_file


def load_observations(root: Path):
    receipt = verify_chain(json.loads((root / "receipt.json").read_text(encoding="utf-8")))
    tables = {}
    for name, sha in receipt["outputs"].items():
        path = root / f"{name}.csv"
        if sha256_file(path) != sha:
            raise ValueError(f"Experiment 交接内容不符：{name}")
        tables[name] = pd.read_csv(path, dtype={"target_id": str, "source_id": str, "region": str}, float_precision="round_trip")
    return tables, receipt
