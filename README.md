# CS336 — Language Modeling from Scratch（Spring 2026）

本项目用于学习 Stanford CS336 2026 春季课程。

- [课程官网与课表](https://cs336.stanford.edu/#schedule)
- [官方讲义仓库](https://github.com/stanford-cs336/lectures)
- 下载日期：2026-11-17
- 官方仓库版本：`de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e`

## 讲义目录

官方已发布的第 1–17 讲保存在 `lectures/`。其中 9 讲使用 Python 讲义，8 讲使用 PDF；Python 文件同时包含讲解文字和示例代码。

| 讲次 | 主题（官网原文） | 本地讲义 |
| --- | --- | --- |
| 1 | Overview, tokenization | [lecture_01.py](lectures/lecture_01.py) |
| 2 | PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) | [lecture_02.py](lectures/lecture_02.py) |
| 3 | Architectures, hyperparameters | [lecture_03.pdf](lectures/lecture_03.pdf) |
| 4 | Attention alternatives and mixture of experts | [lecture_04.pdf](lectures/lecture_04.pdf) |
| 5 | GPUs, TPUs | [lecture_05.pdf](lectures/lecture_05.pdf) |
| 6 | Kernels, Triton | [lecture_06.py](lectures/lecture_06.py) |
| 7 | Parallelism | [lecture_07.py](lectures/lecture_07.py) |
| 8 | Parallelism | [lecture_08.pdf](lectures/lecture_08.pdf) |
| 9 | Scaling laws | [lecture_09.pdf](lectures/lecture_09.pdf) |
| 10 | Inference | [lecture_10.py](lectures/lecture_10.py) |
| 11 | Scaling laws | [lecture_11.pdf](lectures/lecture_11.pdf) |
| 12 | Evaluation | [lecture_12.py](lectures/lecture_12.py) |
| 13 | Data (sources, datasets) | [lecture_13.py](lectures/lecture_13.py) |
| 14 | Data (filtering, deduplication, mixing, synthetic data) | [lecture_14.py](lectures/lecture_14.py) |
| 15 | Mid/post-training (SFT/RLHF) | [lecture_15.pdf](lectures/lecture_15.pdf) |
| 16 | Post-training — RLVR | [lecture_16.pdf](lectures/lecture_16.pdf) |
| 17 | Alignment — multimodality | [lecture_17.py](lectures/lecture_17.py) |

课表中的第 18、19 讲为嘉宾讲座，截至下载时未提供讲义链接。

## 前两讲阅读入口

Python 讲义按函数组织，直接打开文件即可阅读。官网还提供老师上课时使用的逐步展示页面：

- [第 1 讲：官网展示](https://cs336.stanford.edu/lectures/?trace=lecture_01)
- [第 2 讲：官网展示](https://cs336.stanford.edu/lectures/?trace=lecture_02)
- [第 2 讲：录像对应版本](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)

这三个展示页面的原始记录也保存在 `lectures/var/traces/`。

目前已看完前两讲，待梳理的两处内容可以从以下位置开始：

1. **第 1 讲的课程整体 overview**：`lecture_01.py` 中的 `course_syllabus()`，以及其后的 `basics()`、`systems()`、`scaling_laws()`、`data()`、`alignment()`。
2. **第 2 讲末尾的深度学习内容**：`lecture_02.py` 中的 `gradients_basics()`、`gradients_flops()`、`deep_network()`、`optimizer()`、`train_loop()`，随后是梯度累积与激活检查点。

`lectures/README.md` 为上游使用说明；同目录还保留了讲义的公共辅助文件和环境配置。文件来源及校验信息见 `materials-source.json`。
