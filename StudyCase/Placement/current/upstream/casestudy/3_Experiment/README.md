# 3_Experiment

本阶段只读已封存的 Generator 场与 assignment，按登记坐标生产逐站/逐候选观测、逐区域指标、准入与退化状态。
它不出图、不做显著性检验，也不训练或重生成场；统计在 [4_Analysis](../4_Analysis/README.md) 另起一条链。
信息权限与统计单位见 [docs/contracts.md](../../docs/contracts.md)。

## 编号链

每国一套 `01–06`，与 `2_Generator` 同形态：`.py` 生产，`.ipynb` 展示。单元由 `sglib/experiment/registry.py` 定义，
按内容链回执判定 `DONE`；06 是叶清单哈希门。

| 文件 | 单元 | 说明 |
|---|---|---|
| `01_preflight.ipynb` | `preflight.{connection, planning_pool}` | 固定候选池的接入预审与固定选址池；只展示表 |
| `02_observe.py` | `observe.<region>` | 每区域七个观测族（reconstruction / correction / sweeps / allocator / connection / context；UK 八区另有 T1）；`--region` 选子集，`--backend hpc` 只准备（`PREPARED`） |
| `03_planning.py` | `planning.<region>` | 每区域 field × seed 的选址与定容坐标；`--limit` 可分批续跑 |
| `04_bounds.py` | `bounds.c6` | C6 条件上界 |
| `05_defense.py` | `defense.support` | 登记的 defense 表（固定负荷、接入图、AU PV / NZ 库存 / NL 构造子集） |
| `06_audit.ipynb` | `audit` 与门 | 核对 expected/produced 坐标并写 `audit.json`；`run_gate` 逐叶回比叶清单 |

国家目录：[1_UK](1_UK/)、[2_AU](2_AU/)、[4_NL](4_NL/)、[5_NZ](5_NZ/)。

## 配置与登记

- `general/experiment.toml`：共享协议与权威路径。
- `general/registrations.json`：各单元类型全局登记的科学参数；`<国家目录>/registrations.json`：国别登记。
  回执中的科学参数必须等于登记值，单元才算 `DONE`。
- `<国家目录>/<cc>.toml`：区域、代表区、T1 区域与参照、defense 输入。
- `schemas/`：观测与指标表的结构合同（含 `C3/metrics.toml`）。

## 结果根与上游

- 正式根 `results/3_Experiment/<国家目录>`，由 `_closures/` 封存为只读。`SG_PROFILE=smoke` 写到 `results/_smoke`，
  或用 `SG_RESULTS_ROOT` 指定新根。视图与门报告写到 `results/_views/.../3_Experiment/<国家目录>`。
  06 在门之前写一次 `setup/`（配置原件、提交号、环境、回执代码投影核对，合同第 9 节）；只读根按自己的 `setup/config` 判状态。
- 上游对象由入口注入（`upstream=` 映射：DataOverview 交接与 Generator 交接）；`sglib.experiment` 不导入其他阶段。

C7（分辨率变体）已于 2026-09-08 整体移出主链，待重新设计；其数据 defense 仍在 `defense` 单元内。

## CIVD 本地再生与历史更正证据

`rerun_civd.py` 负责配置与编排（orchestration）。活动实现（active implementation）分别位于
`sglib.generator.civd_extension`（输入物化）、`sglib.experiment.civd_observations`（观测与独立均分校验）、
`sglib.analysis.civd_comparison` / `civd_downstream`（统计与科学不变量）、`sglib.report.civd_audit`（完整性审计）。
案例通过公开的 `stage.load_upstream()` / `stage.with_upstream()` 注入交接对象（handoff）；缓存属于单个执行上下文，
同一进程中的不同结果根互不共享交接缓存。

2026-09-14 的 `civd_step4_four_country_20260914` 更正已于 2026-09-18 并入正式根：
`results/3_Experiment`、`4_Analysis`、`5_Report` 与四国 Generator 的 `civd/` 保存该轮正式产物。
历史上 CIVD Step 4 缺失导致簇编号被当作站点编号；原错误产物与更正证据保留在
`results/_archive/superseded_by_civd_correction/`，活动代码不读取该历史目录。

最终 `refactor_20260922_r2` 是已更正实现的结构重构再生（structural refactor regeneration），不构成新的科学更正事件。
原模型、学习需求场、正式回执及封存保持原貌；新产物进入独立可写根。
英国／澳大利亚重新物化冻结 CIVD 数组，荷兰／新西兰按同一容量加权规则本地重算；四国都沿用原坐标系、容量列、
候选场与输入散列。继承产物保留原生成身份；当前代码生成的 CIVD、观测、分析、统计比较和报告记录新回执。
已有正式 C3 回执中的更正范围决定“上次实现是否有效”，比较表不会把已更正输入再次标记成错误实现。

完成重构和临时根冒烟测试（smoke test）后，在仓库根执行：

```powershell
$resultRoot = "results/_releases/refactor_20260922_r2"
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py prepare --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py observe --country uk --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py observe --country au --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py observe --country nl --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py observe --country nz --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py analyze --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py validate --results-root $resultRoot
```

- `prepare` 要求新根，复制未受影响的产物与原回执，跳过所有 `setup` 和 `_closures`；不提前变为只读。
- 四国 `observe` 可以独立执行。恢复执行前核对现有回执、所有输出散列和当前代码身份。
- `analyze` 再生四国 C3/support、跨国综合和受影响报告，并对全部注册候选进行四分配器比较。
- `validate` 首先核验已完成的下游产物；验证中断后可直接恢复，保持已有数值节点和回执。它重读原始候选 NPZ、原回执、站点顺序和坐标，独立检验每个需求场的均分、总量守恒、指标、门控和父回执链接；此步骤不产生封存标记。
- 完整训练／推理只读验收保存为候选根的 `inherited_model_verification.json`（原任务、回执与实际输出散列）。
- 提交全部本任务修改并生成正式设置快照（setup snapshot）后，才执行下列 `seal`。每个阶段／国家根的快照必须使用当前干净提交和 `formal` 配置；冒烟快照不能用于正式封存。

```powershell
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py snapshot --results-root $resultRoot
./.venv/Scripts/python.exe -X utf8 casestudy/3_Experiment/rerun_civd.py seal --results-root $resultRoot
```

`seal` 仅写候选根内的 Generator、Experiment、Analysis、Report 封存及 `correction_release.json`，并绑定代码提交与各设置快照散列。
原正式根不被覆盖，不再写入全局 `results/CIVD_CORRECTION.json` 指针。
所有步骤都在本机运行，不提交高性能计算（HPC）训练或推理作业。

数值验收（numerical validation）保留原五项 C3 主对比；非 CIVD 观测使用相对／绝对容差 `1e-12` 验证后保留原表格数值，
T1 几何使用 `rtol=1e-10, atol=1e-8`，独立 CIVD 站级校验使用 `rtol=1e-12, atol=1e-10`，总量校验使用 `atol=1e-8`。
CIVD 簇内均分（within-cluster equal allocation）未定义唯一站点空间足迹，因此其 T1 交并比（intersection over union, IoU）仍标记为不可评估。

代码身份（code identity）按角色名称与规范化抽象语法树（normalized AST）计算；模块位置和符号定位信息不参与散列。
纯源码搬迁不改变投影；上游注入、调用名称和审计行为调整会改变受覆盖函数的投影。输入物化与观测的新身份不要求重新训练模型。
原有重复的观测投影调用由公共 `observation_code()` 统一；审计直接使用该接口。
本轮前后投影差异、继承身份和本地新产物范围记录在本次变更与封存说明，历史回执保持不变。

`refactor_20260922` 的第一份封存保留原貌。最终 r2 补齐 Windows 原生 JSON 换行兼容测试：普通核心写入保留 LF，学习工作器的排他写入及 AU 紧凑 JSON 保留原生换行。r2 在全新临时根完成冒烟验证后重新执行本地再生，最后以新的干净提交封存。
