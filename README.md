# Mini-Align-LLM

一个面向大模型实习面试准备的完整训练项目。项目目标不是复现超大规模模型，而是从工程和研究两个角度，搭建一套小型但完整的 LLM 对齐训练系统，覆盖 `SFT`、`Reward Design`、`PPO / GRPO`、`Distillation` 与基础评测闭环。

这个项目适合用于：

- 学习小型指令模型训练流程
- 理解 `SFT -> RL -> Distill` 的完整链路
- 熟悉 `PPO / GRPO` 这类对齐算法的工程实现
- 为大模型 research / 算法实习准备项目经历

## 1. 项目目标

本项目希望解决这样一个问题：

> 如何从一个小型预训练语言模型出发，逐步完成监督微调、奖励建模、强化学习对齐与模型蒸馏，最终得到一个可评测、可展示、可部署的小型对齐模型？

围绕这个目标，项目拆成 5 个核心阶段：

1. 基础模型接入与推理
2. 指令数据构建与 `SFT`
3. `Reward` 设计与采样
4. `PPO / GRPO` 对齐训练
5. Teacher-Student 蒸馏与评测

最终交付物包括：

- 一个独立的小型 LLM 对齐训练仓库
- 一套 `SFT` 训练脚本
- 一套 `PPO / GRPO` 训练脚本
- 一套 `Reward` 计算模块
- 一套蒸馏训练模块
- 一套推理与评测脚本
- 一份实验结果与案例分析

## 2. 项目整体流程

整个项目的工作流如下：

```text
选择小型基础模型
-> 构建指令数据
-> SFT 训练
-> 基础评测
-> 设计 reward
-> 对 prompt/sample 打分
-> PPO / GRPO 对齐训练
-> 对齐后评测
-> 生成 teacher 数据
-> student 蒸馏训练
-> 效果 / 延迟 / 资源对比
-> Demo 与 README 展示
```

如果从模型角色来理解，流程是：

```text
Base Model
-> SFT Model
-> Aligned Model (PPO / GRPO)
-> Distilled Student Model
```

如果从数据流来理解，流程是：

```text
Raw Instruction Data
-> Processed SFT Data
-> Policy Samples
-> Reward Scores
-> Preference / RL Batches
-> Teacher Outputs
-> Distillation Data
```

## 3. 实现目标

本项目不追求参数规模，而追求训练链路完整、实验逻辑清楚、结果可解释。

### 3.1 SFT 阶段目标

- 让基础模型具备稳定的指令跟随能力
- 支持标准的 `instruction/input/output` 数据格式
- 支持训练集 / 验证集划分
- 支持 response-only loss
- 能输出可对比的生成样例

### 3.2 Reward 阶段目标

- 为生成结果定义“好回答”的标准
- 支持规则型奖励
- 支持模型型奖励
- 支持多个 reward 组合与加权
- 支持 reward 分布分析

### 3.3 PPO / GRPO 阶段目标

- 实现对齐训练最小闭环
- 能够对同一输入采样多个候选输出
- 能根据 reward 更新策略模型
- 对比 `PPO` 与 `GRPO` 的训练差异
- 观察 `KL`、reward、输出质量的变化

### 3.4 蒸馏阶段目标

- 使用对齐后的 teacher 模型生成高质量数据
- 训练更小的 student 模型
- 比较 teacher / student 的效果差异
- 比较推理速度、显存占用与参数量

### 3.5 评测与展示目标

- 固定 benchmark prompt 做横向对比
- 输出定量评测结果
- 展示典型成功案例与失败案例
- 让整个项目具备清晰的面试展示价值

## 4. 推荐仓库结构

```text
mini-align-llm/
├── README.md
├── requirements.txt
├── configs/
│   ├── sft.yaml
│   ├── ppo.yaml
│   ├── grpo.yaml
│   └── distill.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── scripts/
│   ├── prepare_sft_data.py
│   ├── prepare_pref_data.py
│   └── build_distill_data.py
├── src/
│   ├── models/
│   │   ├── policy_model.py
│   │   ├── value_model.py
│   │   └── tokenizer.py
│   ├── data/
│   │   ├── sft_dataset.py
│   │   ├── pref_dataset.py
│   │   └── collators.py
│   ├── trainers/
│   │   ├── sft_trainer.py
│   │   ├── ppo_trainer.py
│   │   ├── grpo_trainer.py
│   │   └── distill_trainer.py
│   ├── rewards/
│   │   ├── rule_reward.py
│   │   ├── model_reward.py
│   │   └── reward_mixer.py
│   ├── eval/
│   │   ├── generate.py
│   │   ├── metrics.py
│   │   └── case_study.py
│   └── utils/
│       ├── logger.py
│       ├── checkpoint.py
│       └── seed.py
├── train_sft.py
├── train_ppo.py
├── train_grpo.py
├── train_distill.py
└── infer.py
```

## 5. 文件目录说明

### 根目录文件

- `README.md`
  项目说明、实验流程、运行方式与结果展示。

- `requirements.txt`
  项目依赖列表。

- `train_sft.py`
  `SFT` 阶段训练入口。

- `train_ppo.py`
  `PPO` 对齐训练入口。

- `train_grpo.py`
  `GRPO` 对齐训练入口。

- `train_distill.py`
  蒸馏训练入口。

- `infer.py`
  统一推理入口，支持加载不同阶段模型。

### `configs/`

- `sft.yaml`
  `SFT` 训练相关超参数，如学习率、batch size、max length、save strategy。

- `ppo.yaml`
  `PPO` 训练配置，如 rollout batch、clip range、kl coef、sampling temperature。

- `grpo.yaml`
  `GRPO` 训练配置，如 group size、relative reward 设置、采样配置。

- `distill.yaml`
  蒸馏训练配置，如 teacher path、student path、distill loss 权重。

### `data/`

- `raw/`
  原始指令数据、原始偏好数据、外部收集数据。

- `processed/`
  清洗、模板化后的训练数据。

- `eval/`
  固定评测集、测试 prompt、标准答案或规则文件。

### `scripts/`

- `prepare_sft_data.py`
  将原始 instruction 数据处理为 `SFT` 格式。

- `prepare_pref_data.py`
  构建对齐训练用的采样 / 偏好数据。

- `build_distill_data.py`
  使用 teacher 模型生成蒸馏数据。

### `src/models/`

- `policy_model.py`
  策略模型封装，负责加载底座模型、生成输出、前向传播。

- `value_model.py`
  `PPO` 中可选的 value 分支或独立 value 模型实现。

- `tokenizer.py`
  tokenizer 加载、模板编码与解码。

### `src/data/`

- `sft_dataset.py`
  `SFT` 数据集定义。

- `pref_dataset.py`
  偏好 / 采样数据集定义。

- `collators.py`
  padding、mask、labels 构造逻辑。

### `src/trainers/`

- `sft_trainer.py`
  `SFT` 训练循环与验证逻辑。

- `ppo_trainer.py`
  `PPO` rollout、reward 计算、advantage 更新与优化。

- `grpo_trainer.py`
  `GRPO` 组内采样、相对奖励计算与策略更新。

- `distill_trainer.py`
  teacher-student 蒸馏训练逻辑。

### `src/rewards/`

- `rule_reward.py`
  基于规则的奖励，如格式正确率、长度惩罚、重复惩罚等。

- `model_reward.py`
  基于模型打分的奖励，如相似度、分类器分数或其他质量分数。

- `reward_mixer.py`
  多 reward 组合与加权逻辑。

### `src/eval/`

- `generate.py`
  批量生成脚本。

- `metrics.py`
  自动评测指标计算。

- `case_study.py`
  典型样例与失败案例分析。

### `src/utils/`

- `logger.py`
  日志记录与训练信息输出。

- `checkpoint.py`
  checkpoint 保存与恢复。

- `seed.py`
  随机种子与复现实验设置。

## 6. 技术路线

为了兼顾可实现性和面试价值，推荐这条技术路线：

### 6.1 基础模型选择

推荐从小型开源模型开始：

- `Qwen2.5-0.5B`
- `TinyLlama`
- `distilgpt2`

原则是先保证：

- 能在单卡或小资源环境下训练
- 能快速做实验
- 能把重点放在对齐训练流程而不是算力本身

### 6.2 数据格式

推荐统一使用如下结构：

```json
{
  "instruction": "请总结下面这段文本",
  "input": "......",
  "output": "......"
}
```

训练时将其模板化，例如：

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

## 7. 项目分阶段实现流程

### 阶段一：基础模型接入

目标：

- 跑通小型基础模型推理
- 验证 tokenizer、生成逻辑和配置管理

需要完成的工作：

1. 选择一个小型基础模型
2. 封装 tokenizer 和 model loading
3. 跑通单条推理
4. 跑通 batch 推理
5. 写统一推理脚本 `infer.py`

阶段产出：

- 可运行的推理入口
- 模型与 tokenizer 封装

### 阶段二：SFT 数据准备与训练

目标：

- 训练一个具备基本指令跟随能力的模型

需要完成的工作：

1. 清洗 instruction 数据
2. 设计统一 prompt template
3. 构建 `SFT Dataset`
4. 实现 `collator`
5. 只对 response 部分计算 loss
6. 跑 `SFT` 训练
7. 在验证集上评估结果

阶段产出：

- `SFT` 模型 checkpoint
- 训练日志
- baseline 与 `SFT` 对比样例

### 阶段三：Reward 设计

目标：

- 定义什么样的输出是“更好”的输出

需要完成的工作：

1. 定义任务目标
2. 设计规则型 reward
3. 设计模型型 reward
4. 实现多个 reward 的加权组合
5. 统计 reward 分布
6. 检查 reward 是否过于稀疏或偏置

常见 reward 示例：

- 格式正确奖励
- 长度惩罚
- 重复惩罚
- 关键词覆盖奖励
- 参考答案相似度奖励

阶段产出：

- reward 计算模块
- reward 分析结果

### 阶段四：PPO 对齐训练

目标：

- 基于 reward 对模型进行策略优化

需要完成的工作：

1. 准备 policy model
2. 准备 reference model
3. 准备 value model 或 value head
4. 对输入采样多个 response
5. 计算 reward
6. 计算 advantage
7. 加入 KL penalty
8. 执行 PPO 更新
9. 记录 reward / KL / entropy 曲线

阶段产出：

- `PPO` 对齐模型
- 训练曲线
- `SFT` 与 `PPO` 结果对比

### 阶段五：GRPO 对齐训练

目标：

- 理解并实现 group-based 相对奖励优化

需要完成的工作：

1. 对同一输入采样一组候选输出
2. 对组内样本打分
3. 构造相对 reward 或 advantage
4. 执行 `GRPO` 更新
5. 比较 `PPO` 与 `GRPO` 的稳定性与成本

阶段产出：

- `GRPO` 对齐模型
- `PPO vs GRPO` 对比实验

### 阶段六：蒸馏

目标：

- 用更小的 student 模型复现 teacher 的行为

需要完成的工作：

1. 选择 teacher 模型
2. 批量生成蒸馏数据
3. 训练 student 模型
4. 比较 teacher / student 表现
5. 比较参数量、显存、延迟

阶段产出：

- student 模型
- 蒸馏效果报告

### 阶段七：评测与案例分析

目标：

- 让整个项目可被验证、可被讲述、可被展示

需要完成的工作：

1. 准备固定测试集
2. 比较 base / SFT / PPO / GRPO / student
3. 统计自动指标
4. 选取成功案例
5. 选取失败案例
6. 分析失败原因

阶段产出：

- 实验总表
- 可视化图表
- case study 文档

## 8. 运行方式示例

以下命令仅为示例，实际参数根据实现调整。

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

### 8.2 准备 SFT 数据

```bash
python scripts/prepare_sft_data.py
```

### 8.3 训练 SFT 模型

```bash
python train_sft.py --config configs/sft.yaml
```

### 8.4 运行 PPO 对齐

```bash
python train_ppo.py --config configs/ppo.yaml
```

### 8.5 运行 GRPO 对齐

```bash
python train_grpo.py --config configs/grpo.yaml
```

### 8.6 构建蒸馏数据并训练 student

```bash
python scripts/build_distill_data.py
python train_distill.py --config configs/distill.yaml
```

### 8.7 推理测试

```bash
python infer.py --model-path checkpoints/student
```

## 9. 推荐评测维度

这个项目建议至少从下面几个维度做对比：

- 自动指标
  - 格式正确率
  - 关键词覆盖率
  - 重复率
  - 平均 reward

- 人工观察
  - 是否更符合指令
  - 是否更少出现无效输出
  - 是否更稳定

- 工程指标
  - 训练时间
  - 推理延迟
  - 显存占用
  - 参数量

最终建议对比这些模型版本：

- Base
- SFT
- PPO
- GRPO
- Student

## 10. 为什么这个项目适合面试

这个项目的优势在于它不是单纯“跑通一个模型”，而是完整覆盖了大模型岗位里常见的几个核心点：

- 你可以讲清楚 `SFT` 是怎么做的
- 你可以讲清楚 reward 为什么这么设计
- 你可以讲清楚 `PPO` 和 `GRPO` 的区别
- 你可以讲清楚蒸馏为什么重要
- 你可以拿实验结果证明自己不是只停留在概念层面

面试时可以把这个项目概括成一句话：

> 我独立实现了一套小型 LLM 对齐训练系统，从监督微调出发，进一步完成了基于 reward 的 PPO / GRPO 对齐训练，并将对齐后的 teacher 模型蒸馏到 student 模型，最终实现了效果、训练成本和推理效率三者之间的系统性对比。

## 11. 当前实现边界

这个项目的目标是做“小而完整”的训练系统，因此默认有这些边界：

- 不追求超大参数规模
- 不追求超长上下文能力
- 不追求大规模分布式训练
- 重点放在训练链路完整和研究逻辑清楚

这反而更适合实习面试，因为它强调的是你对原理、实现、实验和工程组织的理解。

## 12. 后续可扩展方向

在完成最小闭环后，后续可以继续扩展：

- 支持 `DPO / ORPO / KTO` 等更多对齐算法
- 接入更强的 reward model
- 增加 Web Demo
- 增加实验看板
- 增加多任务评测
- 把该项目和多模态 / diffusion 项目打通

## 13. 建议的项目叙事方式

如果你把这个项目写进简历，建议这样组织表达：

1. 先说明这是一个独立的小型 LLM 对齐训练项目
2. 强调你覆盖了 `SFT -> RL -> Distill`
3. 强调你实现了 `PPO / GRPO`
4. 强调你做了实验对比和失败分析
5. 强调你理解模型效果和部署成本之间的权衡

这样这个项目就会非常适合用于：

- 大模型 research 实习
- 算法工程实习
- 大模型训练 / 对齐方向实习

---
