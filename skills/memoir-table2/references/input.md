# 输入数据

JSON 包含 caption、notes 和 rows。数值在输入时已经是最终展示单位。保留更多有效位，输出时按 decimals 舍入。

```json
{
  "caption": "IR2R navigation performance",
  "decimals": 2,
  "notes": ["TL and NE in meters. Other metrics use a 0–100 scale."],
  "rows": [
    {
      "method": "Memoir + Mamba-2 (4 x 768)",
      "seen": [11.83, 3.20, 72.93, 69.54, 64.47, 60.04],
      "unseen": [10.72, 2.70, 77.27, 75.73, 70.54, 63.35],
      "highlight": true,
      "group_before": false
    }
  ]
}
```

seen 与 unseen 均为六项数组，顺序固定为 TL、NE、nDTW、SR、SPL、t-nDTW。

| 值的形式 | 用途 |
|---|---|
| 72.933803 | 使用全表 decimals 格式化的数值 |
| "72.9" | 需要保留来源展示精度的字符串 |
| {"value": "72.2", "std": "0.2"} | 保留均值与标准差，渲染为 72.2 ± 0.2 |
| {"value": 75.734355, "bold": true} | 显式加粗单元格 |
| null | 指标缺失，输出破折号 |

highlight 用于强调整行。group_before 用于在该行前画分组横线。所有加粗均根据输入决定，脚本按输入处理。

方法名可含换行符，用于过长的结构说明。源论文行需要依附上方基线时，可在方法名中留缩进。把 checkpoint、seed、P、训练划分和来源写入 notes，或在用户确实需要时扩展左侧元数据列。

原始结果字段按以下顺序读取。

```python
keys = ["lengths", "nav_error", "nDTW", "sr", "spl", "t-nDTW"]
seen = [summary["metrics"]["val_seen"][key] for key in keys]
unseen = [summary["metrics"]["val_unseen"][key] for key in keys]
```
