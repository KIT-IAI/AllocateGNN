# 1_DataOverview

本阶段获取公开数据，派生区域、站点、网格和特征，并为每个国家生成清单与交接对象。
可复用的处理代码在 `sglib.dataoverview`；本目录只放编排入口、人工调整的 TOML、展示 Notebook 和产物合同。
数据角色、单位与国家限定见 [docs/contracts.md](../../docs/contracts.md)。

## 配置

- `../config/countries/<cc>.toml`：跨阶段国家事实，是这些事实的唯一来源；本阶段 overlay 不得重复其中的键。
- `<国家目录>/<cc>.toml`：本国数据集与产品台账（下载地址、落地路径、区域及其源顺序）。不登记历史行数或区域数；
  每次重建的实际计数写在该次的 inventory 与 gate 中。
- `general/general.toml`：共享的网格与特征参数，以及每个展示 Notebook 必须包含的
  `[display].required_sections`（source_and_license、schema、coverage_stats、unit_and_vintage、standard_figures）。
- `schemas/*.toml`：正式产物的版本化结构合同。

## 两种执行方式

同一张单元 DAG 有两种执行方式，产物完全相同；编号链只是多了展示。

1. **国别编号链。** 五个国家目录下都有同名的 01–06 六个文件，按编号运行。每个文件只运行 DAG 中属于自己的单元
   （`sglib.dataoverview.stage.run_step`）；前序单元未 `DONE` 时失败并报出缺失的单元 ID；已完成单元 `SKIP`
   并说明依据；`--refresh` 或 `stage.run_step(..., refresh=True)` 强制重做。
2. **可续跑 CLI。** `run_all.py` 覆盖整个登记表，供集群壳与批量重跑使用：

```powershell
python casestudy/1_DataOverview/run_all.py --list
python casestudy/1_DataOverview/run_all.py --phase processing --country uk
python casestudy/1_DataOverview/run_all.py --country au --category station_register
python casestudy/1_DataOverview/run_all.py --country uk --show-config
```

`python -m sglib.dataoverview --list` 是同一 CLI 的包入口。

## 编号链

| 序号 | 文件 | 计算产物 | 展示 |
|---:|---|---|---|
| 01 | `01_download.py` | `data/datasets/1_raw/<cc>/` | 终端 SKIP/RUN/PASS，SKIP 附原因与路径 |
| 02 | `02_regions_stations.ipynb` | `2_derived/<cc>/bplus/` | 台账、schema、区域/站点统计、单位与时点；区域+站点图、分析区域图、首区域放大图 |
| 03 | `03_grid.ipynb` | `2_derived/<cc>/grid_bplus/bundles/<region>/` | 网格参数、bundle schema、逐区域概况；步长/点数图、首区域网格点图 |
| 04 | `04_features.py` | NTL 栅格 `1_raw/<cc>/ntl/*.tif`（GEE，范围取自规范区域）；`2_derived/<cc>/features_bplus/extracted/*.npz` 与回执 | 终端（长时间批处理） |
| 05 | `05_features_overview.ipynb` | 无（只读，要求 04 已完成） | NPZ schema、C/U/Z 与均值覆盖表；土地利用份额、C/U/Z 划分、首区域主导土地利用/C/U/Z/Built-S/NTL 地图与直方图 |
| 06 | `06_inventory.ipynb` | `results/1_DataOverview/<dir>/data_inventory.json` 与交接；首次在其旁写一次 `setup/`（合同第 9 节） | inventory 字段、逐区域覆盖、evidence、产物哈希清单、`setup/` 核对、完成断言 |

各国 02、03 运行的单元：

| 国家 | 02 | 03 |
|---|---|---|
| [1_UK](1_UK/) | `uk.regions.derive`、`uk.substations.derive` | `uk.grid.derive` |
| [2_AU](2_AU/) | `au.stations.derive`、`au.regions.derive`、`au.ledger.derive`、`au.fy2024.derive` | `au.grid.derive` |
| [3_DE](3_DE/) | `de.gva.derive`、`de.regions.derive`、`de.substations.derive` | `de.grid.derive` |
| [4_NL](4_NL/) | `nl.regions.derive`、`nl.stations.derive`、`nl.gate_a.derive` | `nl.grid_skeleton.derive`、`nl.grid.derive` |
| [5_NZ](5_NZ/) | `nz.station_ledger_2024.derive`、`nz.station_sites.derive`、`nz.analysis_regions.derive`、`nz.admission_gate.derive` | `nz.grid.derive` |

01 运行本国无前序的 `<cc>.*.download`；04 依次运行 `<cc>.ntl.download` 与 `<cc>.features.derive`；
06 运行 `<cc>.inventory.overview` 并调用 `handoff.load_bundle`。

五个国家的 Notebook：

- 1_UK：`1_UK/02_regions_stations.ipynb`、`1_UK/03_grid.ipynb`、`1_UK/05_features_overview.ipynb`、`1_UK/06_inventory.ipynb`
- 2_AU：`2_AU/02_regions_stations.ipynb`、`2_AU/03_grid.ipynb`、`2_AU/05_features_overview.ipynb`、`2_AU/06_inventory.ipynb`
- 3_DE：`3_DE/02_regions_stations.ipynb`、`3_DE/03_grid.ipynb`、`3_DE/05_features_overview.ipynb`、`3_DE/06_inventory.ipynb`
- 4_NL：`4_NL/02_regions_stations.ipynb`、`4_NL/03_grid.ipynb`、`4_NL/05_features_overview.ipynb`、`4_NL/06_inventory.ipynb`
- 5_NZ：`5_NZ/02_regions_stations.ipynb`、`5_NZ/03_grid.ipynb`、`5_NZ/05_features_overview.ipynb`、`5_NZ/06_inventory.ipynb`

以 UK 为例（其他国家替换目录名）：

```powershell
.\.venv\Scripts\python.exe casestudy/1_DataOverview/1_UK/01_download.py
.\.venv\Scripts\jupyter.exe execute casestudy/1_DataOverview/1_UK/02_regions_stations.ipynb
.\.venv\Scripts\jupyter.exe execute casestudy/1_DataOverview/1_UK/03_grid.ipynb
.\.venv\Scripts\python.exe casestudy/1_DataOverview/1_UK/04_features.py
.\.venv\Scripts\jupyter.exe execute casestudy/1_DataOverview/1_UK/05_features_overview.ipynb
.\.venv\Scripts\jupyter.exe execute casestudy/1_DataOverview/1_UK/06_inventory.ipynb
```

图全部写入 `results/1_DataOverview/<dir>/figures/`（`02_*.png`、`03_*.png`、`05_*.png`），Notebook 源码保持
`outputs=[]`。`general/00_data_matrix.ipynb` 从五国台账重新生成 `general/data_matrix.md`；它不依赖任何国别 inventory，
可随时运行。`data_matrix.md` 是登记过的生成产物，不要手工编辑。

**SKIP 与 BLOCKED 的依据。** 静态下载看 `landing_<dataset>.json` 回执，GEE 看 `acquisition_receipt.json` 的
`reused` 标志，派生产物看各自的合同；没有回执的文件标为 provenance unverified。NTL 单元需要环境变量
`GEE_SERVICE_ACCOUNT_KEY_PATH`（可写在仓库根 `.env`），缺失时报 `BLOCKED`，已落地的栅格不受影响。

## 国别差异

**UK。** 16 个分析区（ITL2，伦敦合并），覆盖 85 行 ITL3 源；`evaluation_scope = four_task_chain`。

**AU。** 12 个分析区（`loc_key`），SA3 源。FY2024 需求是任务真值，FY2009 只作历史来源。

**DE。** Börde 单区，34 行 Gemeinde（`Name`）源；`evaluation_scope = dataoverview_only`，任何 Generator、Experiment、
Analysis 或 Report 上下文都不得消费。

**NL。** 仅限 D7 设备登记所代表的 Liander 服务区。正式目标是每个 CBS 2025 `buurt` 一个按容量加权的伪站，
源是其上级 CBS `wijk` 多边形；设备行只作血缘与诊断证据。`capacity_basis = nominal_normal_state`，不得称为 firm 或 N-1。
执行顺序为 `regions → stations → gate_a → grid_skeleton → grid → features → inventory`。`gate_a.json` 是独立的训练前准入记录；
`engineering_admission.json` 在不读 OSM/GHSL/NTL 特征的情况下生成，解析冻结的共享工程准入权威，必须为 `ADMITTED`
才能提交训练。分析区域确定性生成：过小的设备分层并入共享 PDOK 边界最长的邻居，其余分层沿 RD New 最长轴递归中位切分，
wijk 源分配到最近叶质心；冻结通过共享硬门的最小预登记 14–16 区树。这一过程不读取任何模型或评价输出。

**NZ。** 冻结范围是 core-9：只含 Vector Lines、Orion NZ 与 Wellington Electricity。正式时点组合为 2024 实测年峰值与
2024 Installed Firm Capacity，连接 2025 Commerce Commission 站点几何与 2023 Stats NZ Census/SA2 特征；2026 披露字段是
预测原型，禁止进入正式真值路径。适配器写出站段血缘台账和独立的站址表（每次重建的行数记在 `audit/nz_gate.json`）；同址站段的峰值与
容量求和并标记 `noncoincident_conservative`，严格 N-1 稳健性只在站址聚合后评估。

NZ 正式重跑不读取研究期物化结果作为输入。在联网的本机或 HAICORE 登录节点落地全部官方输入：

```powershell
python casestudy/1_DataOverview/5_NZ/acquire_fresh_sources.py
```

`--refresh` 重新下载全部对象；`--verify-only` 做离线字节与分页审计。该命令流式下载两份 Commerce Commission Parquet 与两份
determination，先发现 ArcGIS `OBJECTID` 再抓取 Stats NZ/Census 分页，并写出 `data/datasets/1_raw/nz/fresh_sources_manifest.json`；
缺少完整清单时正式 NZ 产品运行拒绝派生。2024 D6 过滤条件与语义行身份记录在同一回执中。落地后，普通产品运行器重建站段、
站址目标、SA2 源和九个登记区域，下游直接使用这份重建结果。重建只检查内部一致性与合同语义（只含三家配电商、2024 年数据、
站址覆盖全部站段、三层需求守恒、区域与登记一致等），不与早期研究版本的表格或计数比对，数据更新带来的漂移是允许的。
内部守恒不能证明原始记录完整，所以重建同时写出三张对账表：D6 2024 核心配电商的每条原始站段及其纳入/排除原因
（`audit/core9_section_reconciliation.csv`，纳入集合必须等于站段台账）、每个已落地 SA2 的归属或排除原因
（`audit/core9_source_reconciliation.csv`），以及候选 SA3 被已落地 SA2 覆盖的比例（`audit/core9_candidate_sa3_coverage.csv`）。
汇总与检查写入 `nz_gate.json` 的 `reconciliation` 与 `checks`。当前正式派生根中的 `nz_gate.json` 生成于这项对账之前，
它的哈希被 NZ inventory 引用，下次正式重建 NZ 时才会带上对账。

## 交接给 Generator

NL 与 NZ 的训练前证据按 fail-closed 设计：Gate 回执与准入状态通过类型化 DataOverview 交接传递，
Generator 正式输入会拒绝任何未 `PASS` 的必需 Gate。本地结构冒烟回执可以使用明确标注的特征替身，但从不授权正式提交；
这些输出会标明哪些字段来自真实重建数据、哪些是合成的轻量 OSM/GHSL/NTL 替身，不是正式 DataOverview 产品。
