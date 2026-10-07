# SpatialGranularity

本项目把区域级用电统计按可审计的空间权重降尺度到规则网格与变电站，再在同一批候选场上
评估站级重建（Reconstruction）、选址与定容（Planning）以及接入成本等下游任务。
覆盖五国：UK、AU、NL、NZ 走完整四任务链；DE 只做数据总览。

这是唯一的工程总入口。数据角色、单位、状态和运行身份等接口约定见
[docs/contracts.md](docs/contracts.md)；各阶段的操作步骤见下文的阶段 README。

## 目录

```text
spatialgranularity/
├── README.md                  工程总入口（本文件）
├── docs/contracts.md          当前生效的工程与科学接口合同
├── sglib/                     唯一活动代码包（各阶段算法、执行与展示）
├── casestudy/                五阶段入口：编号脚本/Notebook、TOML 配置、登记与 schema
│   ├── config/                跨阶段国家档案（countries/*.toml）与共享准入权威
│   ├── 1_DataOverview/        数据获取、派生、网格、特征与国别清单
│   ├── 2_Generator/           分配结构、候选需求场、训练/推理、物化与交接
│   ├── 3_Experiment/          观测、选址定容、上界与 defense 表
│   ├── 4_Analysis/            国别统计与跨国描述性综合
│   └── 5_Report/              跨国图表渲染与审计
├── tests/                     公共测试（见 tests/README.md）
├── data/                      数据登记与本地数据根（见 data/README.md）
├── results/                   科研产物（不入库）
├── private/                   可选本机运维与科研资产（不随检出提供）
├── pyproject.toml             sglib 打包元数据
├── requirements.txt           环境依赖与安装说明
└── pytest.ini                 测试配置
```

论文、答辩、投稿与作者论证材料不在本次工程维护范围内。主包、默认测试和正式流水线不依赖项目外写作目录。

## 环境

Python 3.13，Windows + CUDA 12.6 为主要环境；依赖版本以 `requirements.txt` 为准。

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

没有 GPU 时，torch 与 torch-scatter 改装 CPU 轮子。NTL 下载需要 Google Earth Engine 服务账号：
复制 `.env.example` 为 `.env`，令 `GEE_SERVICE_ACCOUNT_KEY_PATH` 指向 `data/secrets/` 下的 JSON（不入库）。
原始数据、派生数据和结果都不随代码分发，获取方式见 [data/README.md](data/README.md)。

所有案例命令都在仓库根目录执行。可编辑安装（editable install，`-e`）让入口始终加载当前检出的代码，适合科研重构与审计。
运行数据由 `CountryPipelineContext` 的运行根控制；AU 的 `au_registry` 属于包资源（package resource），按包位置读取。
安装位置不再决定运行数据目录；案例配置和已落地数据仍由调用方提供。

## 五阶段入口

每个阶段、每个国家各有一条编号链。按编号顺序执行：前序单元未完成时直接失败（fail-closed），
已完成单元跳过并说明原因。长时间批处理写成 `.py`，轻计算与展示写成 `.ipynb`；
Notebook 源码始终保持 `outputs=[]`。

| 阶段 | 国家 | 编号链 | 说明 |
|---|---|---|---|
| 1_DataOverview | UK AU DE NL NZ | 每国 01–06，另有 `general/00_data_matrix.ipynb` | [README](casestudy/1_DataOverview/README.md) |
| 2_Generator | UK AU NL NZ | 每国 01–07 | [README](casestudy/2_Generator/README.md) |
| 3_Experiment | UK AU NL NZ | 每国 01–06；CIVD 更正重跑入口 `rerun_civd.py` | [README](casestudy/3_Experiment/README.md) |
| 4_Analysis | UK AU NL NZ + 跨国 | 每国 01–03，跨国 01–02 | [README](casestudy/4_Analysis/README.md) |
| 5_Report | 跨国 | 01–02 | [README](casestudy/5_Report/README.md) |

入口读取两个环境变量：`SG_PROFILE`（默认 `formal`）与 `SG_RESULTS_ROOT`。正式结果根
`results/<阶段>/` 已由 `_closures/` 封存为只读；自检或重生请用 `SG_PROFILE=smoke`
或 `--profile smoke --results-root results/_smoke/<新名称>`。图表与门报告写入 `results/_views/`。
2026-09-22 结构重构的最终发布根为 `results/_releases/refactor_20260922_r2/`；完成状态、继承输入与新生成结果的区别见
[封存合同](docs/contracts.md#2026-09-22-结构重构与独立封存)，最终代码提交记录在该根的 `correction_release.json`。

测试入口见 [tests/README.md](tests/README.md)。独立示例见 [examples](examples/README.md)，默认写入临时目录。

## 私有运维边界

`private/` 是被 Git 忽略的可选本机目录，不随公共源码检出提供。其中：

- `private/hpc/`：可选的 HAICORE 集群站点壳；本机已配置时，操作说明位于 `private/hpc/README.md`。
  站点、账户和凭据配置由作者单独提供；
- `private/tests/`：站点壳测试，用 `python -m pytest private/tests` 单独运行；
- `private/campaigns/`、`private/allocator_*`、`private/archive/`、`private/storm_output/`：
  独立科研实验包与调研数据，各自带有来源说明和哈希清单，按资产整体保留。

公共代码、公共测试和本地执行不依赖这些可选文件。集群执行需要另行配置私有站点壳。
HPC 只是显式选择的后端：`--backend hpc` 只准备任务并报告 `PREPARED`，不提交、不代表完成。

## 未实施事项

以下事项仍然有效，但都**尚未实施**，不得当作现行合同引用。

1. **输入合同 v2（尚未实施）。** 目前 DataOverview 把源统计区、候选站和评价目标三种身份绑在一起，
   且绑定发生在分块与分配之前。AU 的源范围由 FY2009 可用站所在 SA3 反推（悉尼漏掉 Carlingford、
   Parramatta 两个 SA3）；NZ 的站表先按正峰值、正 firm capacity 和安全等级筛选，再由站点所在 SA3
   决定源域（63 个 D5 站位被排除，Selwyn District East 因此整体消失，940 条服务面记录未入表）；
   Generator 用同一个站表既作 proximity 上下文又作分配候选池；Experiment 把评价目标与候选池视为同一集合，
   且工程准入的最小格点门会把合法的零接收外围候选判为失败。UK、NL 没有这种反向绑定。
   v2 需要分别声明八份清单：source_catalog、candidate_locations、evaluation_targets、computation_blocks、
   allocation_domains、source_budget_reference、weighter_context、service_areas，按 ID 连接且不假定行序相同。
   VD、CIVD、IDR 的数学规则不变。只拆清单、只改评价集合或只扩候选池时不需要重训；补齐源、网格或公共特征时
   重建图并复用原 checkpoint 推理，此时必须冻结并登记坐标 StandardScaler、`*_percent` 旧分母、
   插补规则和新源所属的旧计算块；改变模型输入含义、监督目标或训练折就属于另一项训练实验。
   新产物必须另立版本、另行封存，旧 release 保持不动。只需要论文中的敏感性结论时，不必实施 v2；
   要把完整覆盖结果作为正式章节结果或新发布版本，或希望新国家、新年份不再继承这种耦合时，才需要实施。
2. **C7 重新设计。** C7（分辨率变体）已于 2026-09-08 整体归档：代码在
   `private/campaigns/006_hpc_r1/c7/`，产物在 `results/_archive/006_c7/`，尚未立新计划。
3. **SYN 表名登记。** `sglib/report/tables.py` 仍硬编码六张综合审计/限制表
  （`claim_coordinate_audit`、`claim_evidence_status`、`inference_resolution_audit`、`coverage_corrections`、
   `target_count_audit`、`limitations_registry`），尚未登记到 `casestudy/5_Report/general/report.toml`。
4. **历史结果目录的保留边界（2026-09-18 已定）。** `results/` 只冻结最终正式结果一批；其余各轮、旧流水线的
   `_releases/013-*`、`_staging/013-*`、`v3`、`v3_final`、诊断与冒烟根整件移入 `results/_archive/`，只作记录，
   代码不得读取（合同第 9、10 节）。断掉的 `results/v3_final/2_Generator` 目录联接已删除。
