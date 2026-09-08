# Point-in-Time Research

股票与 ETF 研究中的时点数据、实验隔离和序列验证内核。从持续维护的多频研究平台提取实际实现，聚焦一个问题：**结果看起来不错时，怎样确认它没有使用未来信息、混入账户状态，或误读时间相关性？**

## 运行一个完整案例

Python 3.10+：

```sh
python -m pip install -r requirements.txt
python -B -m pytest -q
python -B demo.py --out demo-output
```

入口使用明确标注的合成输入，输出成员快照审计、CSV、滚动窗口报告及失败分类。无需行情账户或模型服务。检查 `demo-output/case.json`、`coverage.csv` 和 `validation/summary.md`，即可追到源码。

## 源码阅读路线

| 问题 | 实现 | 可复核行为 |
| --- | --- | --- |
| 历史时点能看到哪些标的？ | [point_in_time_universe.py](qplatform/point_in_time_universe.py) | 只选决策时点之前的快照；过期和未覆盖日期单独报告 |
| 特征与研究输入有没有越界？ | [research_guard.py](qplatform/research_guard.py) | 未来字段、账户隔离、成交可观测性和重复研究来源的检查 |
| 结果是否依赖某一个起点？ | [sequential_validation.py](qplatform/sequential_validation.py) | 重叠多起点窗口、日历分段和极佳日期敏感性 |
| 时间相关性如何影响结论？ | [dependence_statistics.py](qplatform/dependence_statistics.py)、[time_series_diagnostics.py](qplatform/time_series_diagnostics.py) | HAC、分块重采样、序列及波动记忆诊断 |

工程取舍和反例见 [case studies](docs/case-studies.md)。源实现与公开适配见 [provenance](docs/source-provenance.md)。

## Engineering context

This is an extracted research subsystem, not a new toy backtest written for this repository. The original platform also manages data refresh, research programs and execution observations; those account-facing integrations are deliberately separate. Core module names and implementations are retained so the code remains traceable.

Development uses Codex-assisted implementation and iterative engineering. Contract checks inspect supplied evidence; a boolean declaration or clean feature name alone cannot prove a strategy causal. Demonstration returns are synthetic, not trading performance.

## License

[MIT](LICENSE). NumPy, pandas, PyArrow and pytest are separately installed dependencies; their licenses remain with their projects. No broker implementation or proprietary market dataset is redistributed.
