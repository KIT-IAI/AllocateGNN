"""Freeze one verified recompute run (plan section 5.6) and verify the frozen copy.

    python scripts/paper2_freeze.py freeze --run-id <run_id>
    python scripts/paper2_freeze.py verify --run-id <run_id>

``freeze`` requires the Step B receipt PASS, re-checks every recorded output hash,
copies ``recompute/<run_id>`` to ``frozen/<run_id>`` (refusing an existing target),
writes PROVENANCE.md and SHA256SUMS, and marks every frozen file read-only.
``verify`` re-hashes the frozen tree against SHA256SUMS and regenerates the claim
ledger from the frozen files into a temporary folder, requiring byte identity. The
verification record goes to ``manifests/``; the frozen tree is never written again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def provenance(backup: Path, run: Path, run_id: str) -> str:
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    release = json.loads((backup / "inputs/release_r2/correction_release.json").read_text(encoding="utf-8"))
    closures = sorted((backup / "inputs/release_r2").glob("*/_closures/*.json"))
    freeze = cfg["input_freeze_receipt"]
    p = cfg["protocol"]
    lines = [f"# PROVENANCE — `{run_id}`", "",
             "本目录是margin-criterion修订稿正文的唯一数值来源。冻结后不得修改；任何更正须另起run_id。", "",
             "## 上游身份", "",
             f"- 发布：`{release['correction_id']}`（{release['scope']}），代码提交`{release['code_commit']}`。",
             f"- 代码归档：`{receipt['code']['upstream_zip']}`，sha256 `{receipt['code']['upstream_zip_sha256']}`（可由`git archive {release['code_commit'][:7]}`逐字节复现，见输入冻结回执）。",
             f"- Generator closure sha256（发布记录）：`{release['generator_closure_sha256']}`。",
             "- 发布封存文件（备份内路径与sha256）："] + \
            [f"  - `{c.relative_to(backup).as_posix()}` `{sha256(c)}`" for c in closures] + \
            [f"- 活动输入冻结回执：`{freeze['path']}`，sha256 `{freeze['sha256']}`（结论PASS）。",
             f"- 本运行实际读取的输入与哈希：`inputs_manifest.csv`（sha256 `{receipt['inputs_manifest_sha256']}`）。", "",
             "## 负荷口径：固定X，不用λ", "",
             f"- 本文固定X ∈ {{{', '.join(f'{x:g}' for x in p['loads_mw'])}}} MW，主分析{p['main_load_mw']:g} MW（D1），预算不随X变化。",
             "- 上游C4按X = λ·(10 km参考容量中位数)构造情景（λ ∈ {0.25, 0.5, 1}），并以`loss/X`为指标；"
             "该λ情景、`ZERO_X`状态与`loss/X`均不进入本文。上游九项Holm族只出现在差异表的族说明中。",
             "- 英国成本`C_y = 430,000 £/MVA · max(0, G_y + X − F_y) + T_y`，`T_y = X_MW·1000·t_y·AF`，"
             f"AF = (1−1.035^−20)/0.035 = {cfg['annuity_factor_full_precision']!r}（全精度；14.21只作显示）。"
             "澳洲`c = 1, T ≡ 0`，PF=1 MVA等价量，不是货币成本。", "",
             "## 距离实现分工（D6）", "",
             f"- 接入邻域、尺度曲线与接入候选：{cfg['distance_implementation']['connection_and_scale']}。",
             f"- 选址与定容：{cfg['distance_implementation']['siting_and_sizing']}；本目录`tasks/`为r2 C4逐种子产物的导出，未重算。", "",
             "## 费率版本（D4、D5）", "",
             f"- 工作簿`{cfg['tariffs']['source']['workbook']}`，sha256 `{cfg['tariffs']['source']['sha256']}`。",
             f"- T9（主）：{cfg['tariffs']['source']['T9']}。",
             f"- T25（仅英国R=10 km、X=300 MW敏感性）：{cfg['tariffs']['source']['T25']}。",
             "- 分区：`inputs/connection_pricing/dno_zones_20240503.geojson`（14个GSP组＝TNUoS需求分区）；点在面内判定，"
             "2个海岸评价位置按最近分区处理并逐条登记（`uk_tariff_mapping.csv`；D12，2026-09-25拍板；不进入任何入选或oracle集合）。", "",
             "## 统计版本规则", "",
             "- D9：尺度曲线的数值零为区域平均MAE ≤ 1e-9·D_r；LU为零时百分比不定义，胜负平依据判零后的配对误差。",
             "- D10：Voronoi等效半径 = 各区站点最近邻距离中位数的地区中位数 ÷ 2。",
             "- D11：表3接入行主读法用全部有效区域，D2合格区作为并列敏感性行。",
             "- D13：澳洲不报告Π，只报告端点绝对值。",
             "- D14：零界／零后悔判定用相对容差1e-9·max(1, max C)。", "",
             "## 协议要点", "",
             f"- 候选：每区{p['candidate_count']}个行序等步评价位置，κ = {p['shortlist_k']}，{p['tie_rule']}。",
             f"- Ref资格（D2）：{p['ref_rule']}；适用RQ1九分布面板与依赖Ref的接入端点，不用于前三任务、尺度曲线与预算域。",
             f"- 预算：{p['budget_domain']}；η ∈ {p['etas']}，{p['quantile_method']}。",
             f"- 零界判定：{p['zero_tolerance']}；违反判定：{p['violation_tolerance']}。",
             "- 统计：区域为推断单位；GNN三种子区内先平均再与LU配对；精确符号翻转；每国四模块Holm；"
             "区域配对percentile bootstrap（B=10,000，PCG64 seed 42）。见`stats/stats_meta.json`。", "",
             "## 目录", "",
             "- `config.json`、`inputs_manifest.csv`、`code/`、`run.log`、`checks.json`、`receipt.json`：步骤B登记单元`paper2_fixed_load_connection`。",
             "- `candidates/`、`*_controls|cost_oof|scale_station|scale_grid|adequacy|ref_eligibility|region_support.csv`、`uk_tariff_mapping.csv`、`uk_t25_sensitivity.csv`：步骤B产物。",
             "- `tasks/`：重建、选址、定容逐区逐种子值（两种匹配口径）。`stats/`：本文统计。`ledger/claim_ledger.csv`：数值登记表。",
             "- `SHA256SUMS`：除自身外全部文件的sha256。", ""]
    return "\n".join(lines)


def freeze(backup: Path, run_id: str) -> None:
    src, dst = backup / "recompute" / run_id, backup / "frozen" / run_id
    if dst.exists():
        raise SystemExit(f"{dst} exists; frozen runs are never rewritten")
    receipt = json.loads((src / "receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "PASS":
        raise SystemExit("Step B receipt is not PASS")
    for rel, meta in receipt["outputs"].items():
        if sha256(src / rel) != meta["sha256"]:
            raise SystemExit(f"{rel} changed after the Step B receipt")
    stats_receipt = json.loads((src / "stats/receipt.json").read_text(encoding="utf-8"))
    for rel, meta in stats_receipt["outputs"].items():
        if sha256(src / rel) != meta["sha256"]:
            raise SystemExit(f"{rel} changed after the statistics receipt")
    if not (src / "ledger/claim_ledger.csv").is_file():
        raise SystemExit("ledger missing")
    shutil.copytree(src, dst)
    for path in files_under(src):
        rel = path.relative_to(src)
        if sha256(path) != sha256(dst / rel):
            raise SystemExit(f"copy mismatch {rel}")
    (dst / "PROVENANCE.md").write_text(provenance(backup, dst, run_id), encoding="utf-8")
    sums = [f"{sha256(p)}  {p.relative_to(dst).as_posix()}" for p in files_under(dst)]
    with open(dst / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as stream:  # LF, so `sha256sum -c` works directly
        stream.write("\n".join(sums) + "\n")
    for p in files_under(dst):
        os.chmod(p, stat.S_IREAD)
    print(f"frozen {len(sums)} files + SHA256SUMS -> {dst}")


def verify(backup: Path, run_id: str) -> None:
    dst = backup / "frozen" / run_id
    here = Path(__file__).resolve()
    listed = {}
    for line in (dst / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    present = {p.relative_to(dst).as_posix() for p in files_under(dst)} - {"SHA256SUMS"}
    bad = [rel for rel, d in listed.items() if not (dst / rel).is_file() or sha256(dst / rel) != d]
    extra, missing = sorted(present - set(listed)), sorted(set(listed) - present)
    writable = [p.relative_to(dst).as_posix() for p in files_under(dst) if os.access(p, os.W_OK)]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-X", "utf8", str(here.parent / "paper2_ledger.py"), "--root", str(dst), "--out", tmp], check=True)
        regenerated = sha256(Path(tmp) / "claim_ledger.csv")
    frozen_ledger = listed["ledger/claim_ledger.csv"]
    record = {"schema": "paper2_frozen_verification_v1", "run_id": run_id, "verified_utc": datetime.now(timezone.utc).isoformat(),
              "sha256sums_sha256": sha256(dst / "SHA256SUMS"), "files_listed": len(listed), "hash_mismatches": bad,
              "unlisted_files": extra, "missing_files": missing, "writable_files": writable,
              "ledger_regenerated_sha256": regenerated, "ledger_frozen_sha256": frozen_ledger,
              "ledger_regenerates_identically": regenerated == frozen_ledger,
              "ledger_script_sha256": sha256(here.parent / "paper2_ledger.py")}
    record["conclusion"] = "PASS" if not (bad or extra or missing or writable) and record["ledger_regenerates_identically"] else "FAIL"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = backup / "manifests" / f"frozen_{run_id}_verification_{stamp}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=1))
    if record["conclusion"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("freeze", "verify"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    args = ap.parse_args()
    (freeze if args.action == "freeze" else verify)(args.backup.resolve(), args.run_id)


if __name__ == "__main__":
    main()
