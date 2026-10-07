# 2_Generator

本阶段从 DataOverview 交接构建分配结构和候选需求场：静态场、学习场的训练与推理、候选物化、参数扫描、IDR/CIVD 分配，
最后审计并生成交给 Experiment 的类型化交接。可复用算法与执行器在 `sglib.generator`（执行 `stage`、展示 `overview`、
跨阶段交接 `handoff` 与内容链辅助 `chain`）；本目录只放编号入口、配置、权威文件和产物合同。
数据角色、IDR 合同与运行身份见 [docs/contracts.md](../../docs/contracts.md)。

## 配置与权威

- `../config/countries/<cc>.toml`：跨阶段国家事实；国别 overlay（`<国家目录>/<cc>.toml`）只写区域、任务组、执行开关
  （如 `execution.civd_enabled`）、准入检查（工程准入合同、可用状态等）和 IDR 预算 `idr.b_tv`，不含算法覆盖，
  也不登记历史行数或目标数。
- `general/generator.toml`：共享的训练种子、特征列、GPM、校正、GNN/MLP 配置与六类扫描取值。TOML 是人工调整的配置。
- 注册表、科学矩阵与 IDR 文件是版本化科学权威，不并入 TOML：`general/candidate_registry.json`、
  `general/training_task_matrix_scientific_v2.csv`（现行科学矩阵）、`general/training_task_matrix.csv`
  （不可改的 G1-r1 混合容器记录）、`general/idr_mainline_contract.json`。
- 调度资源只放在所选 HPC 站点下，不绑定科学身份；运行时文件系统根只在 rebase 时提供。
- `schemas/*.toml`：静态场、checkpoint、推理场、候选索引、CIVD 与 IDR 等产物的版本化结构与身份合同。

候选注册表覆盖 UK、AU、NL、NZ：物化 38 个正式候选与 3 个仅 QA 的 Uniform-additive 身份。

## 编号链

每国一条 01–07 链，按顺序执行；未完成的前序会被拒绝。

| 步 | 入口 | 工作 |
|---|---|---|
| 01 | `01_inputs.ipynb` | DataOverview 交接、冻结输入包与输入概览 |
| 02 | `02_static.ipynb` | Proximity、assignments、Uniform、GPM 与公开活动场 |
| 03 | `03_train.py` | 训练；显式选择 HPC 后端时只准备任务索引 |
| 04 | `04_infer.py` | 训练校验、推理与推理校验 |
| 05 | `05_materialize.ipynb` | 候选族、最终索引与六类扫描 |
| 06 | `06_allocators.ipynb` | IDR-fixed 与 IDR-matched；Voronoi 变体分区图。CIVD 层按 `general/generator.toml` 的 `[civd]` 协议物化（四国启用），不展示 |
| 07 | `07_audit_handoff.ipynb` | 完整审计、类型化交接与产物哈希 |

四个国家目录各有同名的七个入口：[1_UK](1_UK/)、[2_AU](2_AU/)、[4_NL](4_NL/)、[5_NZ](5_NZ/)。DE 不进入本阶段。

`run_all.py` 是薄编排入口，从 TOML/JSON/CSV 权威发现国家、候选族、任务组、扫描与依赖。除非用 `--unit` 指定单个精确单元，
CLI 会展开前序：

```powershell
python casestudy/2_Generator/run_all.py --country uk --profile smoke --results-root results/_smoke/my_run
python casestudy/2_Generator/run_all.py --country uk --profile formal --step static
```

## 结果根与视图

- 默认结果基目录：formal 为 `results/`（国别输出在 `results/2_Generator/<国家目录>/`，与 `results/1_DataOverview/` 对应），
  smoke 为 `results/_smoke`，preflight 为 `results/_preflight`；`--results-root` 覆盖基目录，其下固定为 `2_Generator/<国家目录>/`。
- 编号 Notebook 与脚本读取 `SG_PROFILE`（默认 `formal`）与 `SG_RESULTS_ROOT`。不设环境变量时目标是正式根；
  快速自检用 `SG_PROFILE=smoke`（或 `--profile smoke`），写到 `results/_smoke/`。
- smoke 保留全部权威任务坐标，只用前四个配置区域，训练两个 epoch；不使用合成特征替身也能完成物化与扫描，
  结果始终标为 `profile=smoke`。
- `results/2_Generator/_closures/` 中被识别的封存文件使所选根只读；缺失或无效的完成证据直接报错。07 的新审计视图写在根外。
  07 在门之前写一次 `setup/`（配置原件与权威、提交号、环境、回执代码投影核对，合同第 9 节）；只读根按自己的 `setup/config`
  判状态并读取交接。
- 所有图与展示用哈希清单写入 `results/_views/2_Generator/<国家目录>/`（正式根）或 `results/_views/<根>/2_Generator/...`。
  Notebook 源码保持 `outputs=[]`。

## 状态

`PENDING`、`PREPARED`、`DONE`、`INVALID`。`PREPARED` 表示任务索引已存在但完成证据不全；`INVALID` 表示标记无法解析或与合同不符。
每个 `SKIP` 都附原因。

## 可选 HPC 后端

公共链不需要任何站点配置即可本地训练与推理。`--backend hpc` 准备相同的科学任务坐标，在任务中记录该后端并报告 `PREPARED`；
准备既不提交作业，也不代表完成。站点适配器必须按声明的后端执行已准备任务，保持任务身份与相对输出位置，取回产物与回执，
再运行现有的训练/推理校验函数。下游计算前所有前序必须为 `DONE`；后端观测不能替代科学输入或输出校验。

结果基目录由 `--results-root` 或 profile 明确选择。移动站点适配器不会移动输入数据、改变结果命名空间、改写已封存回执，
也不授权替换不可变的执行快照。站点实现（HAICORE 壳与战役恢复入口）在仓库外的 `private/` 中，
操作说明见 `private/hpc/README.md`；删除它们不影响本地执行、产物校验或 Notebook 展示。

## 国别说明

**UK、AU。** 薄 overlay；IDR 预算 `b_tv = 0.10`，启用 CIVD 层。

**CIVD 层（四国）。** 2026-09-18 起正式接纳 2026-09-14 更正版协议：`general/generator.toml` 的 `[civd]` 登记共享方法
（HDBSCAN 聚类、噪声站成单例、容量/距离加权、簇内均分），working CRS、容量列与容量口径取自国家档案；协议进入科学配置指纹。
CIVD 单元与交接同时接受 `sg_civd_index_v1`（早期物化）与 `sg_civd_extension_index_v1`（记录协议的索引）；
后者的协议必须逐项等于登记值，交接才接受。

**NL。** 只有在 DataOverview Gate A 与工程准入都通过后才准入；overlay 不含算法覆盖，使用共享的静态、学习、推理与物化实现。
只使用 IDR v1（`IDR-fixed`、`IDR-matched`，`B_TV=0.10`）；已撤回的 prefix-continuous IDR v2 禁用，CIVD 层自 2026-09-18 启用。
`smoke.py` 用真实 wijk 源与 buurt 伪站目标表构建小型确定性交接，运行共享静态操作以及 GNN、MLP 两 epoch 的
train→verify→CPU-infer 链；特征数组是明确的替身，每份回执都是 `formal=false`，完整 OSM/GHSL/NTL 特征仍是 HPC 阶段门。

**NZ。** 仅限冻结的 core-9 范围（Vector Lines、Orion NZ、Wellington Electricity）。Generator 真值是站址表；
站段台账只作血缘与敏感性输入，绝不能传给规范分配。`run_smoke.py` 从 core-9 的新鲜重建（官方原始数据）构建交接，再生成刻意轻量的
网格/特征替身，运行静态分配、v1 fixed-IDR 门、图构建和一个两 epoch 的 GNN 坐标；其输出只是 HPC 前的结构证据，不是正式证据。
