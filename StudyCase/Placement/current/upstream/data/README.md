# data

数据层只放数据登记和本地数据根，不放下载或派生实现。DataOverview 的代码在 `sglib.dataoverview`，执行入口与 TOML
在 `casestudy/1_DataOverview`（见其 [README](../casestudy/1_DataOverview/README.md)）。

## 布局

| 路径 | 入库 | 内容 |
|---|---|---|
| `metadata.toml` | 是 | 人工登记的数据集事实（layer、path、country、source、regenerate、license、validation）；同时是仓库根定位标记 |
| `datasets/1_raw/<cc>/` | 否 | 可重新获取的原始数据与落地回执（`landing_<dataset>.json`、`acquisition_receipt.json`） |
| `datasets/1_raw/osm_snapshot_raw/` | 否 | OSM PBF 快照 |
| `datasets/2_derived/<cc>/` | 否 | 派生产物：`bplus/`（区域与站点）、`grid_bplus/bundles/<region>/`（网格）、`features_bplus/extracted/`（特征 NPZ 与回执）、`lineage/`、`audit/` 等 |
| `datasets/_cache/` | 否 | 慢速源的 Parquet/GeoParquet 解析缓存；源文件变化时自动失效，可随时删除 |
| `secrets/` | 否 | GEE 服务账号 JSON 等凭据，也不进入任何自动备份 |

文件名与落地路径由 `casestudy/1_DataOverview/<国家目录>/<cc>.toml` 的数据集/产品台账和 `sglib.dataoverview.processing`
决定；`metadata.toml` 不登记文件清单。下载目录中的 `manifest_*_landing.json` 只记录本机落地的字节，不作运行时门。

## 获取

所有数据都从公开源重新下载，不使用旧字节或旧哈希作为运行门：

```powershell
.\.venv\Scripts\python.exe -m sglib.dataoverview --list
.\.venv\Scripts\python.exe -m sglib.dataoverview --country uk --step download
```

也可以按国别编号链运行 `01_download.py` 与 `04_features.py`。NTL 需要 GEE 服务账号（环境变量
`GEE_SERVICE_ACCOUNT_KEY_PATH`，见根 README）。NZ 的正式输入另用 `casestudy/1_DataOverview/5_NZ/acquire_fresh_sources.py`
落地，并写出完整的 `1_raw/nz/fresh_sources_manifest.json`。

## 来源与权限

每个数据集的来源、引用要求与许可以 `metadata.toml` 中的条目为准。需要特别注意的限制：

- UK 的 GB-PS-info（CC BY 4.0）要求同时引用 Data in Brief 论文与 Zenodo DOI；ONS 数据为 OGL v3。
- AU 的 Ausgrid 站点数据按 NER Rule 5.13A 公布，Ausgrid 保留版权；NSW 位置快照属于数据共享协议，只能发布派生结果；
  由它们派生的站表同样只发布派生结果。
- DE 的 GeoServer 图层来自项目合作方，边界底源为 BKG（dl-de/by-2-0）；Eurostat 数据需署名。
- OSM 为 ODbL，坐标派生使用需署名；VIIRS NTL 为 NOAA 公开数据；GHSL 来自 EC JRC。

## 不随代码分发的内容

`datasets/`、`secrets/` 与 `results/` 都不入库。仓库只提供重新获取和派生它们的代码与登记；在干净检出上复现完整链，
需要先按上文下载数据并提供凭据。合作方或协议受限的数据（DE GeoServer 快照、AU 的 NSW 位置快照）无法由第三方重新下载。

## 历史口径说明

以下口径写在派生代码中，这里给出它们的依据：

- **AU FY2009 历史窗口**（`sglib/dataoverview/processing/derive/au/au.py`）：历史站点表使用 FY2008–FY2010 三年窗，
  主年 FY2009；站-年缺测率超过 10% 记为 `cleaned_out`；单位只接受 MW，出现其他单位直接失败、不静默换算。
  当前任务真值是 FY2024 同年协议，FY2009 只作历史来源披露。
- **DE Börde GVA 重建**（`sglib/dataoverview/processing/derive/de/legacy.py`）：GVA 表从 Eurostat SDMX CSV 重建
  （Eurostat Data Browser 于 2026-02 改版后，旧 xlsx 导出不可复得）；2022 年七组锚定值与论文一致；
  源区域为 34 个 Gemeinde、13 个 UW，Gemeinde 图层含脏数据，按质心过滤。UW 负荷是 GeoServer 活值，重新下载必然漂移。
