# PROVENANCE — `fixedload_20260925_r2`

本目录是margin-criterion修订稿正文的唯一数值来源。冻结后不得修改；任何更正须另起run_id。

## 上游身份

- 发布：`refactor_20260922_r2`（structural_refactor_regeneration），代码提交`69c4008732ed83dfb876020504baacbfa526d1aa`。
- 代码归档：`code/upstream_release_69c4008732ed.zip`，sha256 `a8ef11fad856f8bbb549eeffb7782211f29181bfea9ba59ad50db1d7efa9f150`（可由`git archive 69c4008`逐字节复现，见输入冻结回执）。
- Generator closure sha256（发布记录）：`8b3a6c8c4e771e244f2bba4b536db47a2d52bb24ec05bfefc08baed26eff4ba4`。
- 发布封存文件（备份内路径与sha256）：
  - `inputs/release_r2/2_Generator/_closures/refactor_20260922_r2.json` `8b3a6c8c4e771e244f2bba4b536db47a2d52bb24ec05bfefc08baed26eff4ba4`
  - `inputs/release_r2/3_Experiment/_closures/refactor_20260922_r2.json` `0b1a35100b0f50a09af744afb6d4aac067c5ca2d530bcb52cdda2a6d2a1fc0fa`
  - `inputs/release_r2/4_Analysis/_closures/refactor_20260922_r2.json` `218d25585aed59dbbfc9f96789f84f040b8c748444722c4a6123340cc0f53e50`
  - `inputs/release_r2/5_Report/_closures/refactor_20260922_r2.json` `c8ce42b0fe6e851a6a292508148c156fe1f02a43e5d67d2ba2c6aff6e7e9b2e3`
- 活动输入冻结回执：`manifests/active_input_freeze_20260925T113746Z.json`，sha256 `d2fafa6accc3aae4961fb1462883622ffdf347850b848c1843accf3696946ec5`（结论PASS）。
- 本运行实际读取的输入与哈希：`inputs_manifest.csv`（sha256 `7da5924d0ceb8d6dbb7448b2b4aeb61dd7cf1df621cb607058ff4859ff57e65c`）。

## 负荷口径：固定X，不用λ

- 本文固定X ∈ {100, 300, 500} MW，主分析300 MW（D1），预算不随X变化。
- 上游C4按X = λ·(10 km参考容量中位数)构造情景（λ ∈ {0.25, 0.5, 1}），并以`loss/X`为指标；该λ情景、`ZERO_X`状态与`loss/X`均不进入本文。上游九项Holm族只出现在差异表的族说明中。
- 英国成本`C_y = 430,000 £/MVA · max(0, G_y + X − F_y) + T_y`，`T_y = X_MW·1000·t_y·AF`，AF = (1−1.035^−20)/0.035 = 14.212403301952268（全精度；14.21只作显示）。澳洲`c = 1, T ≡ 0`，PF=1 MVA等价量，不是货币成本。

## 距离实现分工（D6）

- 接入邻域、尺度曲线与接入候选：projected planar distance in the national working CRS (UK EPSG:27700, AU EPSG:7856), cKDTree closed ball。
- 选址与定容：reused C4 products (haversine distance), not recomputed here；本目录`tasks/`为r2 C4逐种子产物的导出，未重算。

## 费率版本（D4、D5）

- 工作簿`inputs/connection_pricing/tnuos_tariffs_2026_27.xlsx`，sha256 `3c608c80ce10e1786a3fe65a60e067b7aa44da6d7ab5ee3432cbe53c60590d9b`。
- T9（主）：sheet T9 rows 4-17, column 'HH Demand Tariff (£/kW)' (2026/27, zero-floored)。
- T25（仅英国R=10 km、X=300 MW敏感性）：sheet T25 rows 5-18, '2026/27 Final' 'Year Round (£/kW)' (unfloored locational)。
- 分区：`inputs/connection_pricing/dno_zones_20240503.geojson`（14个GSP组＝TNUoS需求分区）；点在面内判定，2个海岸评价位置按最近分区处理并逐条登记（`uk_tariff_mapping.csv`；D12，2026-09-25拍板；不进入任何入选或oracle集合）。

## 统计版本规则

- D9：尺度曲线的数值零为区域平均MAE ≤ 1e-9·D_r；LU为零时百分比不定义，胜负平依据判零后的配对误差。
- D10：Voronoi等效半径 = 各区站点最近邻距离中位数的地区中位数 ÷ 2。
- D11：表3接入行主读法用全部有效区域，D2合格区作为并列敏感性行。
- D13：澳洲不报告Π，只报告端点绝对值。
- D14：零界／零后悔判定用相对容差1e-9·max(1, max C)。

## 协议要点

- 候选：每区2000个行序等步评价位置，κ = 20，numpy argsort kind=stable on raw grid-row order。
- Ref资格（D2）：q_r = max_R mean_y|A_ref,R - G_R| / median_y G_R <= 0.25 with every denominator defined (median G > 0)；适用RQ1九分布面板与依赖Ref的接入端点，不用于前三任务、尺度曲线与预算域。
- 预算：all regions of the country with a valid 2000-candidate pool; not Ref-screened; per method/seed/radius; LOO over regions; budget independent of X；η ∈ [0.5, 0.8, 0.9, 1.0]，linear (numpy)。
- 零界判定：D14 (decided 2026-09-25): value <= 1e-9 * max(1, max_y C_y) counts as zero (L_S, bound_S)；违反判定：1e-10 * max(1, L_E, B_E, L_S, B_S) as in upstream conditional_bounds。
- 统计：区域为推断单位；GNN三种子区内先平均再与LU配对；精确符号翻转；每国四模块Holm；区域配对percentile bootstrap（B=10,000，PCG64 seed 42）。见`stats/stats_meta.json`。

## 目录

- `config.json`、`inputs_manifest.csv`、`code/`、`run.log`、`checks.json`、`receipt.json`：步骤B登记单元`paper2_fixed_load_connection`。
- `candidates/`、`*_controls|cost_oof|scale_station|scale_grid|adequacy|ref_eligibility|region_support.csv`、`uk_tariff_mapping.csv`、`uk_t25_sensitivity.csv`：步骤B产物。
- `tasks/`：重建、选址、定容逐区逐种子值（两种匹配口径）。`stats/`：本文统计。`ledger/claim_ledger.csv`：数值登记表。
- `SHA256SUMS`：除自身外全部文件的sha256。
