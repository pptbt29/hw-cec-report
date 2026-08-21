# Independent jury request

## Initial request

你是独立论文 idea 评审者，不参与候选生成。请直接完整读取文件：

`/Users/dazzysy/workspace/project/hw-cec-report/idea-stage/EDGE_TASK_SCHEDULING_CANDIDATE_DOSSIER_20260724_111654.md`

不要接受执行者摘要，不要编辑任何文件。按照文件第 4 节逐项评审 C1–C11，给出每项六维评分与简洁证据，并选择一个首选、一个备选。必须重点攻击首选的新颖性、现实重要性和可证伪性；给出三个严格 go/no-go 条件、应删除变量，以及与最近正式发表工作的 one-sentence novelty delta。如果合并两个候选，必须证明它们共享同一根因而非堆砌。保持中立；“将 RL/Lyapunov/优化器用于问题”不算新颖性。最终只返回评审报告。

## Evidence addendum

定向检索补充了三项必须纳入重合判断的正式工作：

1. *Adaptive Scheduling for Edge-Assisted DNN Serving*, IEEE MASS 2023, DOI 10.1109/MASS58611.2023.00071，已通过调度提升 edge DNN batching opportunity；
2. *EcoServe*, OSDI 2026，已有 macro-instance、adaptive routing 与 mitosis scaling；
3. *JITServe*, NSDI 2026，已有未知请求信息下的 SLO-aware batch composition。

它们分别加大 C1/C2/C11 的重合风险。请勿因而机械淘汰，但必须精确说明剩余差异。

## Follow-up request

请基于两项新发现做一次极简复审，只回答它们是否改变 C2 首选结论，以及为什么：

1. *SHEPHERD*, NSDI 2023：two-level planning/serving，正式化所有可能 batches，在线 BATCHGEN 更新 candidate batch，并用 model-specific batching 最大化 goodput；
2. *PPipe*, USENIX ATC 2025：edge video analytics pooled pipeline、adaptive batching、probe candidate batch sizes。

若 C2 剩余 delta 已不足，请在 C1–C11 中改选；若仍保留 C2，请给出不与这两篇重合的最小 claim。不要编辑文件。

