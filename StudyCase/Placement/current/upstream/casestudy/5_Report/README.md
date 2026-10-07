# 5_Report

本阶段从封存的 `results/4_Analysis` 机械渲染跨国图表，输出七个单元 C1–C6 与 SYN：共 19 张图、18 张稳定 ID 表和
七份内容链回执（content-chain receipt）。它不重算上游，也不生成结论。

## 编号链

先运行 `01_render.py`，再执行 `02_audit.ipynb`。在仓库根、已配置项目环境中：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
./.venv/Scripts/python.exe casestudy/5_Report/01_render.py --profile formal
```

正式产物位于 `results/5_Report`。七个单元都已 `DONE` 时 01 全部 `SKIP`；正式根缺少渲染单元时拒绝写入。
按主张检查可用 `--claim C1 C4 SYN`。重生时使用尚未使用过的临时结果根：

```powershell
./.venv/Scripts/python.exe casestudy/5_Report/01_render.py --profile smoke --results-root results/_smoke/<新轮次>
```

## 登记与实现

- `general/report.toml` 是登记：数据筛选、顺序与配色都从这里读取（`sglib.report.config`）。
- `sglib.report.registry` 定义单元、依赖与 `report.render.<member>.v1` 节点；`stage` 提供编号执行接口；
  `production` 产生内容链回执，其 inputs 只包含实际读取的 Analysis 单元回执摘要；`figures/` 按主张绘图；
  `tables` 负责机械拼接，保留来源表值与完整源表。
- Analysis 由入口注入 `UPSTREAM['status']` 与 `UPSTREAM['tables']`；`sglib.report` 不导入其他阶段。数据接口按需校验所读 CSV
  与 Analysis 回执中的 SHA-256。
- Report 的 `DONE` 只检查回执、节点、登记和输出存在，不重算输出摘要，也不建立叶清单或哈希门。

## 产物

每个单元包含 `figures/<ID>.png|pdf`、`sources/<ID>.csv`、`tables/<ID>.csv` 与 `receipt.json`。SYN 另原样保留六张综合审计/限制表
（表名目前硬编码在 `sglib/report/tables.py`，见根 README 的未实施事项）。C6 的图源表保留全部行，登记筛选只作用于绘图。
PNG 为 180 dpi；PDF 不保存 CreationDate/ModDate。

## 审计

`02_audit.ipynb` 写出的 `audit.json` 核验七个单元、37 个稳定 ID 与回执摘要。`index.md` 只含两张表：稳定 ID 索引，以及由封存的
T-SYN-04 展开的 30 个实验编号与图表的关联。正式根没有审计时，02 把审计与索引写到 `results/_views/5_Report/`；核对并落根后再次
执行 02，应为 PASS。保存 Notebook 时清空 outputs，`execution_count` 保持为 null。

## 作者结论的位置

结论、解释报告和论文反转复核不由本链生成。它们属于作者材料：历史轮次已迁入仓库外的
`../spatialgranularity-writing-archive/private/report/`；新的主观材料写在被忽略的 `private/report/<轮次>/`，文件开头记录
所依据的正式 `audit.json` 的 SHA-256。r1 参照保留在 `results/_archive/rounds/006_report_r1/`（只作记录）。
`02_audit.ipynb` 在可写根上先写一次 `setup/`（配置原件、提交号、环境、回执代码投影核对，合同第 9 节）；只读根按自己的
`setup/config` 判状态。
