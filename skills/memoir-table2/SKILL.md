---
name: memoir-table2
description: 按 Memoir 论文主文 Table II 的版式创建或更新 IR2R 导航结果表，采用 Methods 加 Val Seen/Val Unseen 双分组表头，支持 PNG、矢量 PDF 和 Markdown。用于论文表2风格复刻、多个世界模型结果合表和向基线表添加实验行。
---

# Memoir Table II 排版

将已确认的导航指标排成每个方法一行、Seen 和 Unseen 左右并列的宽表。参考为 Memoir v2 主文 Table II，PDF 第 9 页。附录也有 Table II，主文的 IR2R 表是此 skill 的目标。[参考论文](https://arxiv.org/pdf/2510.08553v2#page=9)

## 结构与样式

- 左列 Methods，右侧为 Val Seen 和 Val Unseen 两个等宽分组，各跨六列。
- 两组指标均依次为 TL ↓、NE ↓、nDTW ↑、SR ↑、SPL ↑、t-nDTW ↑。每个模型一行，对应的检查点保持一致。
- 两层表头。组名加粗居中，组名下各画一条短横线，指标位于其下。方法名左对齐，数值列居中，固定小数精度。
- 使用 Times 风格的衬线字体、白底和紧凑行高。使用横线组织内容，组间留白。指标表头底为双细横线，表底线稍重。方法组之间可加横线。详细尺寸见 [版式说明](references/layout.md)。
- 保留来源的标准差和原始精度。单次测量显示单个数值，缺失值使用破折号。TL、NE 按米，其他四项按 0–100 分制。
- 模型名称过长时增宽方法列或在明确的语义边界换行。caption 和表注在表外换行，数字与单位保持清楚可读。

## 当前项目的默认选择

以下来自用户已确认的 Memoir 表格偏好。用户的新要求优先。

- 表格保留 Methods 及两组指标。ph、th、phi、iw 四列省略。
- 分组通过细横线和行缩进表达。省略 Traditional VLN、默认协议预训练等跨行说明文字。
- 作者方法保留 Memoir 或基于某基线的 +Memoir 名称，移除容易混淆归属的 Ours 字样。基线依赖关系应能从方法名或缩进看清。
- 用户的模型名称用 Memoir + Mamba-2，并在需要区分时附上 2×512、4×768，混合版写为 Memoir + Hybrid（3 Mamba-2 + 1 Attention，768）。明确的用户命名优先。
- 当任务包含原论文基线和用户实验时，可高亮用户实验行，并在表注说明高亮含义。仅比较用户自己的多个版本时，按请求选择高亮目标。逐列最优的加粗由明确的比较范围决定，行高亮含义需要说明。

## 从结果到表格

1. 优先读取结果汇总 JSON 或用户给出的数值，记录来源、检查点、seed、batch、检索上限 P 和 split。按已约定的 train_dev 选模或用户指定的检查点取数。用户要求汇总已有结果时，直接复用结果。
2. 标准化为 [输入格式](references/input.md) 中的 JSON。原字段映射为 lengths→TL、nav_error→NE、nDTW→nDTW、sr→SR、spl→SPL、t-nDTW→t-nDTW。先检查来源的量纲。
3. 同一行的 seen 和 unseen 使用同一个模型身份。数据范围、预算或 P 有差异时，在表注简洁注明。已有均值±标准差和单 seed 结果保持其统计含义。
4. 调用随 skill 提供的渲染脚本。PNG 适合预览，PDF 保留矢量文字与线条，Markdown 适合仓库文档。只输出用户需要的格式。用户只指定绘制表格时，默认 PNG、PDF 和 Markdown。
5. 打开最终 PNG 检查两组表头、指标顺序、箭头、±、中文字体、加粗与分隔线。对照输入核对首末行和全部指标值。若交付 PDF，另用 PDF 阅读器或 pdftoppm 渲染该 PDF 检查裁切与字体。

脚本依赖 Python 3 和 Matplotlib，Markdown 单独输出仅需标准库。优先使用已有可用 Python 环境，必要时在独立环境安装 matplotlib。

```bash
python /path/to/memoir-table2/scripts/render_table.py data.json \
  --output /path/to/results/table-ii --formats png,pdf,md
```

可用 `--font /path/to/font.ttf` 指定覆盖所需字符的字体，用 `--font-size 10` 设置表体字号。中文内容需要可用的中文字体，PDF 渲染后确认实际字形。

自带的 [示例输入](assets/example.json) 使用三种 Memoir 世界模型已保存的六项双集指标，展示输入格式和长方法名处理。执行实际任务时以用户本次结果为准。

## Markdown 输出

标准 Markdown 无法表达合并表头。使用扁平列名，例如 Seen TL ↓ 与 Unseen TL ↓，保留每个模型一行。若目标支持 HTML，可另用 colspan=6 表达两个合并组名。严谨的版式展示使用图像或 PDF。

Markdown 的整行高亮通过粗体呈现。多格式共用表注时，采用“突出显示该实验行”等跨格式表述，或为每个格式单独调整关于底色的说明。
