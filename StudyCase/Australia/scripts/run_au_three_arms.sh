#!/usr/bin/env bash
# ============================================================
# AU 三臂补跑 runner（2026-07-17）：ntl → proximity → fusion 串行
#
# 臂定义（相图三臂落点补齐）：
#   ntl        单 NTL 先验臂     012_train_au_gnn.py    200 epochs λ=0.05
#   proximity  单 Proximity 先验臂 012_train_au_gnn.py  200 epochs λ=0.05
#   fusion     特征融合臂（7 维）  015_train_au_fusion.py 200 epochs base 损失
# 每臂 = 3 seed × 4 fold = 12 次训练。
#
# 纪律（memory 教训，缺一不可）：
#   - MPLBACKEND=Agg          EdgeWeightSolver:455 的 plt.show() 会永久卡死无人值守进程
#   - KMP_DUPLICATE_LIB_OK    Windows 双 OpenMP 运行时冲突
#   - PYTHONIOENCODING=utf-8  控制台中文
#   - 半成品 fold（有 model.pth 但训练日志缺失/不满）先删再跑——
#     否则 012/015 的 assert_training_complete 会 SystemExit 拒跑
#
# 行为：
#   - 幂等：已完整的臂自动跳过（seed 级 kfold_test_*.csv + fold 级产物齐全判定）；
#     未完整的臂交给训练脚本自身的断点续跑逻辑
#   - 失败即退出非零（set -e + trap），不吞错
#   - 日志追加到固定文件 training_console_three_arms.log（内部 exec 重定向）
#   - 臂完成后不做任何评估、不动任何论文/结果登记文件（用户令）
#
# 用法：
#   bash StudyCase/Australia/scripts/run_au_three_arms.sh            # 正式（后台挂）
#   DRY_RUN=1 bash StudyCase/Australia/scripts/run_au_three_arms.sh  # 只报告计划，不训练
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AU_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_ROOT="$AU_DIR/data/processed/training"
LOG_FILE="$AU_DIR/data/processed/training_console_three_arms.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS=(42 123 456)

# 三臂串行顺序：先两个单先验臂（012，共用既有 5 维缓存），后融合臂（015，7 维缓存）
ARMS=(ntl proximity fusion)

export MPLBACKEND=Agg
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# 正式模式：全部输出追加到固定日志（DRY_RUN 保持控制台可见）
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec >> "$LOG_FILE" 2>&1
fi

log() { echo "[$(date '+%F %T')] $*"; }

trap 'log "[FAILED] runner 异常退出（exit=$?）——检查上方最后一臂输出；半成品 fold 下次启动会被自动清理"' ERR

arm_script() {  # 臂 → 训练脚本
    case "$1" in
        ntl|proximity) echo "$AU_DIR/012_train_au_gnn.py" ;;
        fusion)        echo "$AU_DIR/015_train_au_fusion.py" ;;
        *) echo "未知臂 $1" >&2; return 1 ;;
    esac
}

# ── 半成品清理 + 完整性判定（嵌入 python，逐臂调用）──
# mode=clean : 删除"有 model.pth 但训练日志缺失/epochs 不满"的 fold 目录（先删再跑纪律）
# mode=check : 臂完整（3 seed 汇总 CSV + 4 fold 产物齐）→ 输出 COMPLETE，否则 INCOMPLETE
arm_scan() {  # $1=config $2=mode(clean|check|scan)  scan=只报告不删（DRY_RUN 用）
    "$PYTHON_BIN" - "$TRAIN_ROOT" "$1" 200 "$2" <<'PYEOF'
# -*- coding: utf-8 -*-
"""半成品 fold 清理 / 臂完整性判定（runner 内嵌；不触碰任何完整产物）。"""
import json
import shutil
import sys
from pathlib import Path

train_root, config, epochs, mode = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
SEEDS = [42, 123, 456]


def log_complete(fold_dir: Path) -> bool:
    """训练日志在且 epochs 打满（assert_training_complete 同判据）。"""
    log_path = fold_dir / 'model_training_log.json'
    if not log_path.exists():
        return False
    try:
        d = json.loads(log_path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return len(d.get('train_losses', {}).get('total', [])) == epochs


def fold_complete(fold_dir: Path) -> bool:
    """fold 完整 = model.pth + 完整日志 + 三张指标 CSV。"""
    return ((fold_dir / 'model.pth').exists() and log_complete(fold_dir)
            and all((fold_dir / f'{n}.csv').exists() for n in ('rmse', 'mae', 'corr')))


if mode in ('clean', 'scan'):
    n_bad = 0
    for seed in SEEDS:
        for k in range(1, 5):
            fold_dir = train_root / f'seed_{seed}' / config / f'fold{k}'
            model = fold_dir / 'model.pth'
            # 半成品判据：模型快照在，但训练日志缺失/不满（硬中断遗留）
            if model.exists() and not log_complete(fold_dir):
                n_bad += 1
                if mode == 'clean':
                    shutil.rmtree(fold_dir)
                    print(f'  [clean] 删除半成品 fold: {fold_dir}')
                else:
                    print(f'  [scan] 发现半成品 fold（正式运行时将删除）: {fold_dir}')
    print(f'  [{mode}] {config}: 半成品 fold {n_bad} 个')
elif mode == 'check':
    ok = True
    for seed in SEEDS:
        seed_dir = train_root / f'seed_{seed}' / config
        if not all((seed_dir / f'kfold_test_{m}.csv').exists()
                   for m in ('rmse', 'mae', 'corr')):
            ok = False
            break
        if not all(fold_complete(seed_dir / f'fold{k}') for k in range(1, 5)):
            ok = False
            break
    print('COMPLETE' if ok else 'INCOMPLETE')
else:
    sys.exit(f'未知 mode: {mode}')
PYEOF
}

log "════════════════════════════════════════════════════════════"
log "AU 三臂补跑 runner 启动 | DRY_RUN=$DRY_RUN | 臂序: ${ARMS[*]} | seeds: ${SEEDS[*]}"
log "python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
log "训练根目录: $TRAIN_ROOT"
log "════════════════════════════════════════════════════════════"

# ── B4 溯源保全：备份 012 既有 run_manifest.json（只备一次，已存在则跳过）──
if [ -f "$TRAIN_ROOT/run_manifest.json" ] && [ ! -f "$TRAIN_ROOT/run_manifest_b4_original.json" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        log "[计划] 备份 run_manifest.json → run_manifest_b4_original.json"
    else
        cp "$TRAIN_ROOT/run_manifest.json" "$TRAIN_ROOT/run_manifest_b4_original.json"
        log "已备份 B4 原 run_manifest.json → run_manifest_b4_original.json"
    fi
fi

for arm in "${ARMS[@]}"; do
    script="$(arm_script "$arm")"
    log "──────────────────────────────────────────────"
    log "臂 [$arm] | 脚本: $script"

    # 幂等：臂已完整 → 跳过
    status="$(arm_scan "$arm" check | tail -1)"
    if [ "$status" = "COMPLETE" ]; then
        log "臂 [$arm] 已完整（3 seed × 4 fold 产物齐），跳过"
        continue
    fi

    # 半成品 fold 先删再跑（DRY_RUN 只报告）
    if [ "$DRY_RUN" = "1" ]; then
        arm_scan "$arm" scan
        log "[计划] $PYTHON_BIN $script --all --configs $arm --seeds ${SEEDS[*]}"
        continue
    fi
    arm_scan "$arm" clean

    log "臂 [$arm] 训练启动..."
    "$PYTHON_BIN" "$script" --all --configs "$arm" --seeds "${SEEDS[@]}"
    log "臂 [$arm] 训练完成"

    # 完成后复核（防脚本静默早退）
    status="$(arm_scan "$arm" check | tail -1)"
    if [ "$status" != "COMPLETE" ]; then
        log "[FAILED] 臂 [$arm] 训练脚本退出但产物不完整——中止后续臂"
        exit 1
    fi
done

log "════════════════════════════════════════════════════════════"
log "三臂全部完成（或已完整跳过）。按用户令：不做评估、不动论文/结果登记文件。"
log "产物: $TRAIN_ROOT/seed_{42,123,456}/{ntl,proximity,fusion}/"
log "════════════════════════════════════════════════════════════"
