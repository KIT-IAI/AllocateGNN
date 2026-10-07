# 工程与科学接口合同

本文件只写**已经生效**的合同。尚未实施的输入合同 v2 等事项见根 [README](../README.md#未实施事项)，
不能引用为现行规则。数值与成员以机器可读登记（TOML/JSON/CSV）为准；本文件说明它们的含义和边界，
不重抄参数表。操作步骤见各阶段 README。

## 1. 权威位置

| 内容 | 唯一权威 |
|---|---|
| 跨阶段国家事实（范围、CRS、时点协议、单位、站表列、容量口径） | `casestudy/config/countries/<cc>.toml`（`sg_country_profile_v1`，文件名即两字母国家码） |
| 阶段内配置 | 各阶段 `general/*.toml` 与国别 overlay；overlay 不得重复国家档案中的键 |
| 数据集登记与仓库根标记 | `data/metadata.toml` |
| 候选注册表与训练矩阵 | `casestudy/2_Generator/general/candidate_registry.json`、`training_task_matrix_scientific_v2.csv`（现行）；`training_task_matrix.csv` 是不可改的 G1-r1 历史记录 |
| IDR 方法合同 | `casestudy/2_Generator/general/idr_mainline_contract.json` |
| 工程准入 | `casestudy/config/authority/engineering_admission_v3.toml` 与各国 `engineering_admission_v3.toml` |
| Experiment / Analysis 登记 | `casestudy/3_Experiment/general/registrations.json`、`casestudy/4_Analysis/general/registrations.json` 及各国 `registrations.json` |
| Report 登记 | `casestudy/5_Report/general/report.toml` |
| 产物结构合同 | 各阶段 `schemas/*.toml` |
| 受控词表与国家档案读取 | `sglib/core/infra/terms.py` |

版本化的科学权威只能按新版本修订，不得就地编辑既有冻结件或回执。

## 2. 数据角色

- **源（source）**：带区域需求总量的统计区，按国家档案的 `source_key` 标识（UK `ITL3`、AU `SA3`、
  NL `wijk_code`、NZ `SA22023_V1_00`、DE `Name`）。
- **分析区域（analysis region）**：源的分组，是训练折、统计推断和 Experiment 的基本单位；
  区域及其源顺序在 DataOverview 国别 TOML 的 `regions.items` 中登记。
- **站点 / 目标（station / target）**：带观测需求与容量的变电站或伪站；是重建的评价对象，也是分配候选。
  NL 的目标是每个 CBS `buurt` 一个按容量加权的伪站；NZ 的目标是站址（同址站段合并而成），站段台账只作血缘与敏感性输入。
- **网格（grid）**：`2_derived/<cc>/grid_bplus/bundles/<region>/` 下的 B+ 规则网格。
- **特征与 C/U/Z 支撑**：C 为有 OSM 覆盖的格点，U 为无 OSM 但 Built-S>0 的格点，Z 为两者皆无；
  Z 的六个土地利用通道逐位为零。
- **候选场（candidate field）**：Generator 产出的格点需求场；**分配（assignment）**把格点需求交给站点。

## 3. 国家范围与单位

| 国家 | 评价范围 | 需求 / 容量单位 | 容量口径 | 必须保留的限定 |
|---|---|---|---|---|
| UK | 四任务链；16 个分析区（ITL2，伦敦合并），85 个 ITL3 源 | MVA / MVA | `firm_n1` | — |
| AU | 四任务链；12 个分析区，SA3 源 | MW / MVA | `reconstructed_firm_n1` | FY2024 同年协议；FY2009 只作历史来源披露，不得作为任务真值；容量耦合计算按声明的 PF=1 视为 MVA 当量，不是实测功率因数 |
| NL | 四任务链；Liander 服务区，16 个分析区，wijk 源，buurt 伪站 | kW / kW | `nominal_normal_state` | 容量不得称为 firm 或 N-1；伪站由非同时峰值求和 |
| NZ | 四任务链；core-9（Vector Lines、Orion NZ、Wellington Electricity），SA2 源，站址目标 | MVA / MVA | `declared_security_class` | 2024 实测年峰值 + 2024 Installed Firm Capacity；2026 披露字段是预测原型，禁止进入正式真值；同址站段峰值与容量求和并标记 `noncoincident_conservative` |
| DE | 仅 DataOverview；Börde 单区，34 个 Gemeinde 源 | MW / 不适用 | 不适用 | Generator、Experiment、Analysis、Report 必须拒绝 DE |

## 4. 术语（代码标签）

以下是代码与登记中的标签。论文用语须有原论文出处，写作时以归档中的术语表为准，不据代码标签造词。

- **分配器（allocator）**：`VD`（欧氏 Voronoi，规范登记槽）、`IDR-fixed`、`IDR-matched`（IDR v1，
  见第 7 节）、`CIVD`（四国 Generator 单元均启用，协议登记在 `generator.toml` 的 `[civd]`；
  CIVD 判读直接读正式根，见第 10 节）。
- **候选族**：`Uni`（Uniform）、`GPM`（只作标签使用）、`Equal`（保留代码标签；`EqualGrid` 是 Generator 候选，
  `EqualStation` / `EqualRegion` 是 Experiment 内由 DataOverview 上下文直接构造的参考）、`MLP`、`GNN`。
- **辅助信号与校正**：`N` = 夜间灯光，`P` = 真实站点邻近性，`NP` = 两者；`post` / `add` = 乘性 / 加性后校正；
  `fusion*` 为特征融合训练配置。
- **证据层**：`L11` 是四个正式任务的部署合法候选；`L22` 是修订 C/U/Z 支撑上的机制复现；`LX` 是探索性配置；
  `L31` 只用于 Reconstruction，且包含 `L11`。资格状态见 `EligibilityStatus`
  （`ELIGIBLE`、`RECONSTRUCTION_ONLY`、`QA_ONLY`、`INELIGIBLE_*`）。
- **主张**：Analysis 与 Report 的单元 `C1`–`C6` 与综合单元 `SYN`。

## 5. 阶段输入与输出

| 阶段 | 读取 | 写出 |
|---|---|---|
| 1_DataOverview | 公开源、国家档案与 DataOverview TOML | `data/datasets/1_raw`、`2_derived`；`results/1_DataOverview/<dir>/` 的 inventory、图表与交接 |
| 2_Generator | DataOverview 类型化交接（`sglib.dataoverview.handoff`） | `results/2_Generator/<dir>/`：assignment、候选场、checkpoint、推理场、物化索引、扫描、IDR/CIVD、审计与交接 |
| 3_Experiment | 入口注入的 DataOverview 与 Generator 交接 | `results/3_Experiment/<dir>/`：观测、选址定容、C6 上界、defense 表、审计与叶清单 |
| 4_Analysis | 入口注入的已封存 Experiment 表与状态 | `results/4_Analysis/<dir>/` 与 `9_CrossCountry/`：对比、区间、统计族、审计与叶清单 |
| 5_Report | 已封存的 `results/4_Analysis` | `results/5_Report/`：C1–C6、SYN 的图、源表、稳定 ID 表、回执、`audit.json` 与 `index.md` |

`casestudy/1_DataOverview/general/data_matrix.md` 由流水线生成并登记，属于产物，不是人工文档。

## 6. 信息权限

- **阶段隔离**：`sglib` 的各阶段包互不导入；上游对象由编号入口注入（`upstream=` 映射）。
  Allocator 全部位于 Generator；Experiment 不训练、不重生成场；Analysis 不重跑模型、分配器或求解器；
  Report 不重算上游，也不生成结论。
- **推理隔离**：推理前从图中物理删除 TEST 区域的目标、先验与监督张量；学习任务保留 OOF 血缘。
  学习臂使用种子 42/123/456 与四折 OOF；折只用于追溯，不是重复样本。
- **站点信息**：`P` / `NP` 臂只作为已知站点 Reconstruction 的主证据，不进入以站点位置为评价对象的规划任务。
  Siting、Sizing、Connection 只使用 VD 规范槽。
- **分配器输入**：IDR 只能读取匿名站点坐标、由坐标导出的几何量和公开活动场；不得读取站点峰值、容量、电压、
  运营商、真实服务区，也不得在部署区域用评价真值决定是否启用。
- **国家不池化**：国别观测、bootstrap、p 值与 Holm 族永不跨国合并；跨国综合只读国别摘要，
  不重新 bootstrap，不产生联合 p 值或跨国多重校正。
- **主观材料**：结论、解释与论文反转复核不属于流水线产物，只写在被忽略的 `private/report/<轮次>/`
  或仓库外的写作归档中。

## 7. 方法合同

- **IDR v1**：固定 `alpha=-0.5`、Gaussian 核、站点最近邻距离中位数作带宽；先过 G0 可行性门
  （重现规范 VD、正权重、合法标签、质量守恒、无空站），再过 G1：原始 IDR 与规范 VD 的站级公共质量分布
  总变差不超过显式预算 `B_TV`；任一不过即回退规范 VD。`B_TV` 没有隐藏默认值，现行国别登记为 0.10，
  含义是最多允许 20% 的站级质量重分配。G0/G1 只控制合法性与改动规模，不保证收益。
  v2（prefix-continuous）已撤回，任何国家都不得使用。
- **选址候选池**：四国统一使用 C/U 非 Z 可建设格点；请求数
  `M = max(min(max(n_buildable // 100, 300), 1000), 2k)`，uniform KMeans 不读取候选场
  （`random_state=42`、`n_init=10`、Lloyd、`max_iter=300`、`tol=1e-4`，国别 working CRS）；
  质心映射回网格并去重后要求实际 `M ≥ 2k`，否则不准入且不临时扩池；`k` 严格等于实际站点数。
- **扫描**：固定参数扫描保持 `parameter_name / parameter_value` 与 `metric / value` 分离；
  不得在外层 TEST 上选最佳参数。
- **log-MSE 阈值**：旧阈值 `σc/(2σr)` 只作回顾诊断；同时报告均值项、精确恒等式残差和方向不一致数量，
  不把恒等式命中率当作机制证据。

## 8. 统计合同

- **单位与聚合**：误差先按区域×种子计算，再在区域内平均种子，然后做区域配对。静态臂不复制为三个种子。
  禁止先平均三个种子的预测再算非线性指标。
- **两臂水平比较**：同一国、同一登记有效区域集合内，`d_i = a_i − b_i`；主效应为 `mean_i(d_i)`，
  主相对效应为 `Σd / Σb`（本国区域指标之和的比，不是跨国或全站池化）。两者共用同一套 10,000 次配对区域
  bootstrap 索引，取 2.5% / 97.5% 百分位（`method="linear"`）。区域百分比中位数只作描述，不配区间、不检验。
  单个 `b_i=0` 不排除该区域的原生效应；`Σb=0` 时相对效应不可定义；某次抽样 `Σb*=0` 时相对区间记不可评价，
  不补抽、不加 ε。
- **检验**：双侧穷举区域 sign-flip，α=0.05，每国独立 Holm 族（C1 为每国 10 成员族）。区间是逐对比名义区间，
  不能替代 Holm 判定。离散分辨率不足时标 `INFERENCE_RESOLUTION_LIMITED`，保留效应与区间，不写“无效应”。
- **空间相关**：Queen 邻接与登记区域顺序决定贪心配对块；Moran 对种子平均后的区域配对差做 999 次置换、
  双侧 0.05、不调整。只有“Moran 报警且整块 bootstrap 不支持”的单条对比降为描述性，family 不变。
- **坐标状态**：`VALID`、`MISSING_REQUIRED`、`METRIC_NOT_ASSESSABLE`、`INELIGIBLE_BY_DESIGN` 等状态保留在完整库存中；
  缺件阻塞相关主张，不静默删行。证据不足是科学结果，工程缺件是执行失败，两者不混用。

## 9. 运行身份与状态

- **内容链**：每个节点的承诺记录节点 ID、实际消费输入的哈希、科学参数投影哈希与代码投影哈希；
  回执再记录输出哈希。下游唯一的校验是“输入哈希等于上游输出哈希”。实现只有一处：
  `sglib/core/infra/content_chain.py`。调度资源、路径布局、队列、时限和机器绝对路径不进入科学参数，
  最多作为回执中的 `observations`。纯运维变更要区分三层：
  - **数值结果**：不得改变。
  - **科学身份**（输入哈希、科学参数投影、代码投影）：运维字段不进入科学参数；但代码投影覆盖被投影函数（或整个模块）的源码，
    修改这些代码——即使只改路径记录方式——也会产生新的代码身份。
  - **运行溯源**（provenance 文件、`observations`、回执中的输出哈希）：记录方式改变会改变这些文件的字节，从而得到新的回执。
  出现后两种变化时产生新的回执并如实记录原因；不得改写旧封存记录，也不得伪造与旧回执相同的哈希。
- **封存与门**：`results/<阶段>/_closures/` 使正式根只读；DataOverview、Generator、Experiment 与 Analysis
  用叶清单（`sglib/core/infra/leaf_manifest.py`，各阶段 `manifest.py` 适配）在审计后逐叶核对；
  Report 不建叶清单，由 `02_audit.ipynb` 核验七个单元与稳定 ID。
- **设置快照（`setup/`）**：每个结果根（`<阶段>/<国家目录>/`、`9_CrossCountry/`、`5_Report/`）在封存步骤
  （DataOverview 06、Generator 07、Experiment 06、Analysis 03/02、Report 02）写一次 `setup/`
  （`sglib/core/infra/setup_snapshot.py`，`sg_setup_snapshot_v1`）：`setup.json` 记提交号、工作区是否干净及改动文件、
  Python 与平台、依赖清单哈希、各文件哈希，以及用当前代码重算每份回执代码投影的逐条结果（不一致逐项列出，不静默）；
  `config/` 按仓库布局保存本阶段本国实际读取的配置原件（国家档案、阶段 TOML、权威文件、registrations）；`env.txt` 是
  依赖清单；`dirty.patch` 只在 smoke 且工作区不干净时保存，正式封存要求工作区干净。代码版本只记提交号，不打标签。
  `setup/` 是运行溯源，不进入科学身份；它作为一个叶进入叶清单（DataOverview 在 inventory 旁单独核对）。
  **已封存根按自己的 `setup/config` 判定状态和读取交接**：Experiment、Analysis、Report、Generator 的上下文遇到只读根且
  存在 `setup/config` 时，从那里读取登记值与权威，因此 Git 中的 registrations 与权威可以按设计演进而不改变封存单元的状态。
  最终正式结果的 `setup/` 是 2026-09-18 事后重建的（`source = reconstructed_2026-09-18`），其中的私有编排脚本副本在
  `orchestration/`，与回执不一致的代码投影和无法定位的提交在 `setup.json` 中如实标注。
- **结果根**：`SG_PROFILE`（`formal` / `smoke` / `preflight`）选择基目录，`SG_RESULTS_ROOT` 或 `--results-root`
  覆盖；其下固定使用 `<阶段>/<国家目录>/`。视图与首次审计写入 `results/_views/`。
  `results/_archive/` 只作记录（其 `README.md` 与 `manifest.csv` 登记每项的原路径、新路径、原因与日期）：
  结果根解析与 Generator 交接拒绝落在其中的根，`sglib`、`casestudy`、`tests` 不得引用它（架构门检查）。
- **路径**：持久化的回执、清单、日志与指针只记录相对某个根（仓库根或结果根）的 POSIX 路径（`sglib.core.infra.paths.portable_path`），
  读取方按自己的根解析；源码、配置、文档与测试夹具都不写机器相关的绝对路径，由架构门检查。
  早期产物中仍有绝对路径（如 V2 训练任务文件），读取时一律按当前仓库根与结果根重新定位，不按其中的绝对路径直接访问。
- **重建不钉死旧结果**：重新派生的数据产品只检查内部一致性与合同语义，不与早期研究版本的表格、计数或几何逐点比对；
  漂移在允许范围内，下游使用本次重建。配置只登记身份（分析区域、源 ID 及其顺序），不登记某次物化的行数、区域数或
  目标数；每次重建观测到的计数写在该次的 inventory 与 gate 中，架构门禁止配置重新出现这类计数。
- **完整性靠对账，不靠计数**：内部守恒不能证明原始记录完整，所以要逐条对账：
  - NZ 重建为 D6 2024 核心配电商的每一条原始站段写出纳入/排除及全部未通过的规则
    （`audit/core9_section_reconciliation.csv`），纳入集合必须正好等于站段台账；
    每个已落地 SA2 都有归属或排除原因（`audit/core9_source_reconciliation.csv`）；
    候选 SA3 必须被已落地 SA2 完整覆盖（`audit/core9_candidate_sa3_coverage.csv`）。汇总与检查写入 `nz_gate.json`。
  - Generator 冒烟把重建的源 ID 与登记的 `source_key_order` 逐区对账（缺失、未登记、重复），并要求每个站址恰属一个区域。
  - 跨国综合（`sglib/analysis/synthesis.py`）逐区检查非空、粒度比自洽、源与目标总量守恒，并要求每个效应区域都有上下文。
- **身份变化的记录**：科学身份的定义改变时，旧回执不改写，也不在 Git 中登记让旧指纹继续被接受的机制；
  可写根中带旧指纹的输入回执一律拒绝。变化本身只在本文留文字记录：
  2026-09-18 Generator 科学配置投影升为 `sg_generator_scientific_config_v3`，把历史计数
  （`checks.expected_*` 与 NZ `station_contract` 的 `formal_truth_rows`、`section_lineage_rows`）移出身份。
  v2→v3 指纹：UK `fbf7923b…`→`a96d3a4f…`（放回 `expected_n_source_rows=138`、`expected_n_targets=4176`）；
  AU `7669a2d3…`→`9fdabe53…`（放回 34/12/173）；NL `2e2b873d…`→`37c220a4…`（放回 1283/16/5592）；
  NZ `1ae78def…`→`338b5fd1…`（放回 465/9/135、`expected_n_section_lineage_rows=140`、
  `formal_truth_rows=135`、`section_lineage_rows=140`）。同日 `[civd]` 协议登记入投影（见 [2_Generator README](../casestudy/2_Generator/README.md)），四国指纹再次变化；
  正式根的输入回执仍带 2026-09-06 的 v1 指纹，由其 `setup/` 记录这一差异。
- **状态**：`PENDING`（未开始）、`PREPARED`（任务索引已就绪，完成证据不全，例如 HPC 后端只准备未执行）、
  `DONE`（回执、节点身份、登记参数与输出都满足合同）、`INVALID`（标记无法解析或与合同不符，拒绝运行）。
  每个 `SKIP` 必须附原因；DataOverview 缺少 GEE 凭据时该单元报 `BLOCKED`。前序未 `DONE` 时一律失败。
- **Notebook**：源码保持 `outputs=[]`、`execution_count=null`；正式表图只写结果根或 `_views`。

## 10. 冻结批次与验收

- **保留的原正式批次**：`results/1_DataOverview`、`2_Generator`、`3_Experiment`、`4_Analysis`、`5_Report` 是最终正式结果。
  2026-09-18 起 CIVD 更正发布 `civd_step4_four_country_20260914` 并入正式根：`3_Experiment`、`4_Analysis`、`5_Report`
  三层整体取自该发布（自带 `_closures/civd_step4_four_country_20260914.json`），四国 Generator 的 `civd/` 层取自该发布，
  其 Generator 封存放在 `results/2_Generator/_closures/`，并新增一份合并后封存（非 CIVD 节点沿用原回执，CIVD 节点取更正版回执）。
  被取代的旧三层、UK/AU 旧 `civd/` 层与旧指针 `CIVD_CORRECTION.json` 移入 `results/_archive/superseded_by_civd_correction/`；
  原 `delivery_005`、`postprocess_v1` 封存保持原样作历史记录。已封存回执与 provenance 不改写（其中的来源路径是封存时的
  绝对路径，审计器按仓库根重新定位）。更正取代缺少 Step 4 实现时的历史 CIVD 误差、TV 诊断、T1 IoU 与排除结论；
  原五项预注册 C3 对比保留并已核验。CIVD 判读直接读正式根，不再有更正指针。重跑入口见
  [3_Experiment README](../casestudy/3_Experiment/README.md#civd-本地再生与历史更正证据)。
- **Report 验收**：`results/5_Report/audit.json` 与 `index.md`。
- **归档批次**：其余各轮（`v3`、`v3_final`、`_preflight/*`、006/007 各轮、旧流水线的 `manifest.json`、`_releases/013-*`、
  `_staging/013-*`、诊断与冒烟根）在 `results/_archive/` 内只作记录，不冻结；当前项目的任何分析都不得基于归档。
- **私有科研包**：`private/allocator_*`、`private/archive/*` 等保留自己的协议、哈希清单与审计程序。
  其中 r02 包的完成审计按路径核对主项目文件，清理前已因主项目后续改动而无法重跑通过，现作为已完成的历史回执保留。
  `private/archive/coordinate_only_vd_search` 与 `allocator_factorial_closed` 仍引用已退役的 `SpatialGranularity`、`CaseStudy`，
  主环境不再支持运行它们；需要复现时检出旧代码退役前的提交 `0eea7035b508d90e1935a63c156083a983dcd560`，配合原实验包使用。

### 2026-09-22 结构重构与独立封存

本次以 `72c09c6` 的干净工作树为基线执行 S1—S10。新结果根为
`results/_releases/refactor_20260922_r2/`，完成标记为该根的 `correction_release.json`，其中的
`code_commit` 绑定生成正式设置快照和封存时的干净提交。旧正式根、模型、学习需求场、回执、设置和封存不改写。
项目外论文／写作目录按作者本次指示不纳入维护范围。

- **代码身份。** 改动前实际有 11 处生产代码投影调用：包内 6 处、CIVD 案例 5 处；这是盘点结果，不是数量门禁。
  CIVD 迁包后共享观测投影去除一次重复调用。设置快照用显式定位信息重新读取符号，只作验证。
  推理的 30 个角色和规范化抽象语法树保持相同，三个符号迁到字段编码与检查点模块；投影仍为
  `fb42436c07de8edddf086d0bd883b3ed387f3acc13a94cc99b02a04ccfda2be3`。
  Generator 数值、Experiment 数值／规划、Analysis 与 Report 的既有科学投影保持不变。
  CIVD 输入投影由 `347bd819…` 变为 `1802c4f3…`；观测回执的组合投影由 `8f75dfdd…` 变为 `12b50f98…`；
  比较由 `5071d06f…` 变为 `26383694…`；审计由 `113f2f60…` 变为 `90936617…`。
  角色集合保持不变，变化来自公开依赖注入、上下文隔离、封存拒写及如实区分已纠正输入；CIVD 聚类、容量加权、
  簇内均分和既有统计内核保持原规则。完整逐符号比较随本次封存保存，不建立永久身份台账。
- **继承与再生。** 四国 480 项训练和 480 项推理按矩阵、任务、完成回执和实际文件逐项验证；推理验证包含
  `inference_completion.json` 的内嵌内容链回执。继承模型和预测保留原生产身份，不因源码搬迁改写回执。
  本地重新物化 CIVD（UK/AU 复用冻结分配数组，NL/NZ 重算分配），再生四国 allocator/T1 观测、C3/support、
  跨国综合、四分配器比较及 Report C2—C6/SYN。未受影响的规划、边界、defense、其他核心分析与 Report C1
  继承原回执。新回执引用继承输入散列并记录当前投影。原 `target_count_audit.csv` 的固定计数登记结构已由现行生产器的
  运行数据结构替代：共同字段精确比较，新比率、总量和校验状态独立绑定原／新上下文及父回执散列；不把退役计数列写回新结果。
- **验证顺序。** 首先完成公共测试、双数据根隔离、同国双结果根 CIVD、本地 GNN/MLP 两轮训练与推理、旧检查点
  零容差权重读取、NL/NZ 临时根冒烟。随后执行正式候选根 `prepare → observe → analyze → validate`。
  完成所有代码和验证修正并提交后，`snapshot` 使用既有设置快照和叶清单机制保存 14 个阶段根及发布根的干净提交，
  最后 `seal` 核验设置、输入与输出回执并创建新封存。没有提交高性能计算训练或推理作业。
- **定位兼容。** 活动案例目录已改为 `casestudy`。冻结权威、证据、历史任务和旧设置中的 `casestudy2` 字符串按原字节保留，
  读取时由路径定位器寻找实物；缺失的旧 `results/v3_final` 参数定位到当前正式根，不读取归档结果。
  Windows 的退役目录检查按父目录实际条目名称比较，避免大小写不敏感文件系统把 `CaseStudy` 命中新 `casestudy`。
  私有 HPC 活动壳的当前源码路径同步更新；已冻结实验包内的旧路径保持不变，未上传或执行集群命令。

本轮最终封存采用 `refactor_20260922_r2`。第一份 `refactor_20260922`（提交 `6076564`）保持原貌；完成审计发现两项需要修正的兼容／证据细节：
学习工作器的旧 JSON 写入在 Windows 使用原生 CRLF，AU 的紧凑 JSON 同样使用原生行尾；统一写入器须保留这些字节，同时维持核心普通 JSON 的 LF。
回归测试直接以旧 `Path.write_text()` 操作生成参考文件，覆盖非 ASCII 和转义换行，逐字节与散列核验。
第一份封存的 `git_blob_bytes_identical` 使用了归档导出内容作比较；r2 改为直接读取初始提交的原始 Git blob，明确区别 Git 内容与初始工作树字节，不改历史记录。
r2 的验证记录／快照以 `*_receipt_file_sha256` 标记整个 JSON 文件散列；内容链的 `receipt_sha256` 继续表示逻辑回执身份，二者分别核验。
