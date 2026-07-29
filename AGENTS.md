# Project writing conventions

- 所有 Markdown 文档中的数学公式必须使用 Lark 可解析的 LaTeX 分隔符：行内公式写为 `$...$`，独立公式写为 `$$...$$`。
- Markdown 中不要使用 `\(...\)` 或 `\[...\]` 作为公式分隔符。
- 公式内容优先使用 Lark/KaTeX 支持的标准 LaTeX 命令；避免在公式中嵌入复杂中文文本，必要时改写为符号并在正文解释。

# Literature review conventions

- 相关工作和新颖性判断优先采用 CEC/MEC、计算机系统、网络与机器学习领域的公认顶级期刊和会议论文。
- 正式发表版本存在时，优先引用正式版本而不是 arXiv 版本。
- arXiv-only preprint 只作为近期并发风险或补充线索，必须明确标注为 preprint；不能与顶刊顶会论文等权支撑问题重要性、方法有效性或研究共识。
- 低质量论文即使主题高度相关，也不能作为核心论证依据；但若其公开时间较早且机制直接重合，仍应作为 novelty risk 单独记录。
