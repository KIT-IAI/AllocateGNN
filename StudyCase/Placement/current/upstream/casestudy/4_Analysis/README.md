# 4_Analysis

本阶段按登记对已封存的 Experiment 观测做配对比较、关联、区间与统计族，生成主张状态；跨国部分只做描述性综合。
它不重跑模型、分配器或求解器。统计单位、估计量与检验规则见 [docs/contracts.md](../../docs/contracts.md) 第 8 节。

## 编号链

- 每国依次执行 `01_core.py` → `02_support.py` → `03_overview.ipynb`（国家目录：[1_UK](1_UK/)、[2_AU](2_AU/)、
  [4_NL](4_NL/)、[5_NZ](5_NZ/)）。`01_core.py` 支持 `--claim C1 C2` 选子集，前序必须完成。
- 跨国依次执行 `9_CrossCountry/01_synthesis.py` → `02_overview.ipynb`；它需要所选根下四国的 core 与 support 全部完成。
- `DONE` 单元跳过，`INVALID` 单元拒绝运行。

每国有六个核心单元、一个 support 单元和一个审计；跨国有描述性综合与审计。

跨国综合的 `target_count_audit` 逐区对账区域上下文（非空、粒度比自洽、源与目标总量守恒），国家目标数从区域上下文读取，
不再与写死的历史计数比较。**身份变化记录（2026-09-18）。** 这项修改改变了 `synthesis` 单元的代码投影；已封存的
`9_CrossCountry/synthesis` 保持原样（其中的 `registered_target_count` 列是旧规则的历史记录），重新生成时得到新的代码身份。

## 结果根与上游

默认 `SG_PROFILE=formal` 只读 `results/4_Analysis/<国家目录>/`。本地重生使用
`--profile smoke --results-root results/_smoke/<名称>`；Experiment 输入仍来自封存根 `results/3_Experiment/`，由入口注入
已核验的表与状态，`sglib.analysis` 不导入 Experiment。

## 登记

登记来自 r1 的 29 份内容链回执（content-chain receipts）：全局共享值在 `general/registrations.json`，国别值在各国
`registrations.json`；大型坐标表以 SHA-256 投影，并从已登记的区域、方法、种子重建。`sglib.analysis.registry` 核对回执、
节点身份、登记参数和输出是否存在；`stage` 强制编号依赖并跳过 `DONE` 单元。两张 C4 支撑表按数据键确定排序：
区域指标按任务、区域、候选场；匹配汇总按候选登记顺序。封存根保存新链的完整产出，r1 表只作再生性参照与历史档案。
`c1`–`c6`、`paired_inference` 与 `linear_contrasts` 中的统计函数保持不变；`schemas/contrasts.toml` 是对比表的结构合同。

## 概览、审计与门

概览包含对比表、证据状态、C1–C5 森林图预览，以及 C6 上界审计与容忍度表；预览只写入 `results/_views/4_Analysis/`。
正式根与已封存根只读：首次审计和清单先生成到视图目录，核对后由执行流程落根。`sglib.analysis.manifest` 把每个单元
作为一片叶哈希；审计落根后生成清单，门逐叶核对；一份清单覆盖一个国家或跨国目录。
