# 独立本地示例

在仓库根完成 `python -m pip install -e .` 后执行：

```powershell
python examples/local_generator.py
python examples/local_generator.py --country nl
python examples/local_generator.py --country nz
```

默认输出到每次新建的系统临时目录，执行后打印回执（receipt）位置；`--output-root` 指定独立的可写目录。合成示例要求空目录，同一进程能够运行多个独立上下文（context），不会覆盖正式结果。

- `synthetic`：无需本机数据或模型；生成两个 UK 合成源区域与两个站点，经上下文中的原始 CSV 完成需求聚合，显式注入交接，生成八个网格点的静态场并核验质量守恒（mass conservation）。
- `nl`：读取已有 NL Gate A 源区域和伪站；抽取四个区域的少量记录，生成轻量特征，执行静态场、CIVD、IDR、MLP/GNN 各两轮 CPU 训练及推理验证。
- `nz`：读取已核验的 NZ 官方原始数据，重建 core-9 交接；在临时根执行结构化特征和四区域 GNN 两轮 CPU 训练，核验静态场、CIVD、IDR 和工程规模门。

可复用实现位于 `sglib.examples`；[入口脚本](local_generator.py) 与案例 NL/NZ 冒烟工具仅负责参数解析和调用。示例清单明确记录 `formal=false`，局部候选索引（candidate index）明确记录静态候选范围。新回执引用本次物化的输入清单，不借用正式根的清单指纹。

这些示例不提交高性能计算（HPC）作业。轻量网格、代理特征与冒烟模型不进入正式预测；正式 OSM/GHSL/NTL 特征和工程准入（engineering admission）仍按[工程合同](../docs/contracts.md)验收。

NL/NZ 的完成回执附带完整产物清单（artifact manifest）。复用前重新检查实际文件集合、字节数、散列、输入清单、模型完成回执及 NL 推理的内嵌内容链；复用过程不写入文件，也不重新训练或推理。
缺件、内容变化、依赖链不一致，以及只有状态字段或缺少产物清单的旧冒烟回执，均拒绝复用。此时使用新的空输出目录重新执行；旧回执与封存保持原貌。
