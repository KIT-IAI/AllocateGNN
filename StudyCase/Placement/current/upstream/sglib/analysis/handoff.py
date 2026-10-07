"""Analysis 表只读交接；展示层不回读 Experiment 场。"""

import json
from pathlib import Path
import pandas as pd

from sglib.core.infra.content_chain import verify_chain
from sglib.core.infra.hashing import sha256_file


def load_tables(root: Path):
    receipt = verify_chain(json.loads((root / "receipt.json").read_text(encoding="utf-8")))
    tables = {}
    for name, expected in receipt["outputs"].items():
        path = root / f"{name}.csv"
        if sha256_file(path) != expected:
            raise ValueError(f"Analysis 交接内容不符：{name}")
        tables[name] = pd.read_csv(path, dtype={"target_id": str, "region": str}, float_precision="round_trip")
    return tables, receipt
