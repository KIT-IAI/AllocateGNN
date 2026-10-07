# tests

公共测试只覆盖 `sglib` 与 `casestudy`。站点壳测试在被忽略的 `private/tests/` 中，单独运行。

文档门禁（documentation gate）同时检查当前工作树和临时公共检出。临时检出只复制 Git 跟踪文件的当前内容，
不含 `private/`、`results/`、`data/datasets/` 或 `data/secrets/`；保留随检出提供的 `data/README.md` 与 `data/metadata.toml`。
两处都要求全部公共文档从总入口可达，且每个实际本地链接的目标存在。私有运维路径只作为可选路径文本，不作为公共链接。

## 分组

测试模块或具体用例通过 `pytestmark` / `@pytest.mark` 声明工作流类别。
基础设施、数据派生、数值内核、入口合同和已有产物验证分别组织；同一文件中的具体用例能够声明不同类别，收集阶段没有文件名单限制。

| 标记 | 覆盖内容 | 修改哪些代码时运行 |
|---|---|---|
| `produce` | 数据获取、派生、准入、产物写入与哈希、报告渲染、HPC 搬运工具 | `sglib.dataoverview`、`sglib.core.infra`、`sglib.report` |
| `consume` | 训练、推理、交接读取、分配器、规划与统计内核、Experiment/Analysis 消费逻辑 | `sglib.generator`、`sglib.experiment`、`sglib.analysis`、`sglib.core.algorithms` |
| `gate` | 编号链与目录布局、架构边界、叶清单与准入门、设置快照、文档索引与链接 | 入口、目录、登记与合同文件 |

`slow` 描述耗时。`local_data` 标记本机已落地数据依赖；`local_results` 标记已完成冒烟根依赖；
`formal_results` 标记对正式产物的只读核验。它们与工作流标记正交，合成数据测试无需本机结果。

```powershell
.\.venv\Scripts\python.exe -m pytest                 # 全部公共测试
.\.venv\Scripts\python.exe -m pytest -m produce      # 按类别
.\.venv\Scripts\python.exe -m pytest -m "not slow"   # 本地快速反馈
.\.venv\Scripts\python.exe -m pytest private/tests   # 站点壳测试（仅作者本机）
```

## 无数据测试与数据依赖测试

大多数测试使用自足的数值夹具（例如 `test_algorithm_properties.py`、`test_infra_properties.py` 以及各阶段的编号链测试），
在干净检出上就能运行。以下测试需要本机已有的数据或产物，缺失时会跳过或失败，结论只对该产物成立：

- `test_004_nz_core9.py`：在临时目录里从本机已落地的 NZ 官方原始数据（`data/datasets/1_raw/nz/` 及其
  `fresh_sources_manifest.json`）重建 core-9，检查内部一致性、合同语义与原始站段/SA2 对账，不与任何早期研究版本比对。
- `test_generator_existing_artifacts.py::test_generator_registry_and_smoke_handoff`：设置 `SG_GENERATOR_SMOKE_ROOT` 指向一个完成 01–07 的结果根时才运行，否则跳过。
- `test_004_gee_query.py`、`test_004_nl_wijk_buurt.py` 中的部分用例只读本机 `data/datasets/` 或 `results/_smoke/`
  中已有的产物；GEE 用例先把原始栅格复制到临时根，再核验并生成回执；原始目录保持不变。

## 隔离

测试不得写入正式 `results/` 或 `data/`：写产物的测试都使用 pytest 的 `tmp_path` 作为结果根，
Analysis 视图在 `test_007_analysis_sequence.py` 的夹具中重定向到 `tmp_path`。新增测试必须遵守同样的做法，
不要把 `SG_RESULTS_ROOT` 指向正式根。运行时设置 `PYTHONDONTWRITEBYTECODE=1`，并加 `-p no:cacheprovider`
避免在源码树留下缓存。测试夹具同样不写盘符或机器相关的绝对路径，架构门会检查这一点；架构门也禁止配置登记历史计数
（`expected_*`、`formal_truth_rows`、`section_lineage_rows`、`[regions.checks]`）。

重构验收使用：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -m "not local_data and not local_results and not formal_results" -q -p no:cacheprovider
```

`test_learned_engine_refactor.py` 的微型训练与推理均为本地 CPU 计算，输出进入 `tmp_path`。
公共测试调用的 HPC 分支只验证准备状态，不调用集群提交器；不运行私有站点测试来完成本次验收。
