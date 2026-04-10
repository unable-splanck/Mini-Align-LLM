# Mini-Align-LLM Learning Notes

这份文档用于持续记录我在这个项目里的学习过程。
后续每次和项目相关的提问、解释、实验结论，都会继续追加到这里。

## How To Use This File

- 按主题记录，而不是只按时间堆积。
- 每次新增内容时，优先补充“概念 + 代码位置 + 运行链路 + 我的理解”。
- 如果后续实现发生变化，需要同步更新旧结论，避免笔记过时。

---

## 2026-04-10

### Topic: 项目运行逻辑总览

当前项目已经跑通的主线是：

```text
原始数据
-> 数据预处理
-> SFT 训练
-> 保存 checkpoint
-> 加载 checkpoint 推理
```

目前真正已经实现并验证通过的是：

- `SFT`
- `infer`

目前还只是脚手架、还没有进入真实训练逻辑的是：

- `PPO`
- `GRPO`
- `Distill`

### 1. 入口层的职责

这个项目的入口脚本尽量保持简单，它们主要负责：

1. 读取命令行参数
2. 加载配置文件
3. 调用 `src/` 里的核心模块
4. 保存结果或打印日志

例如：

- [train_sft.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_sft.py)
- [infer.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/infer.py)
- [train_ppo.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_ppo.py)

我的理解：

入口脚本像“控制台命令”，真正的训练和数据细节不应该堆在这里，而应该交给 `src/` 目录里的模块处理。

### 2. SFT 训练是怎么启动的

执行下面这条命令时：

```bash
.venv/bin/python train_sft.py --config configs/sft.yaml
```

内部流程如下：

1. [train_sft.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_sft.py) 读取 `configs/sft.yaml`
2. 设置随机种子
3. 调用 `build_sft_trainer(config)`
4. 构建 tokenizer、model、dataset、collator、TrainingArguments、Trainer
5. 执行 `trainer.train()`
6. 保存模型和训练状态到 `checkpoints/sft`

关键代码位置：

- [train_sft.py:10](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_sft.py#L10)
- [src/trainers/sft_trainer.py:19](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/trainers/sft_trainer.py#L19)

### 3. 配置文件在做什么

[configs/sft.yaml](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/configs/sft.yaml) 主要控制三类信息：

- `model`
  - 加载哪个基础模型
- `data`
  - 用哪些训练文件
  - 最大长度是多少
  - prompt template 长什么样
- `training`
  - batch size、学习率、epoch、保存和评估策略

我的理解：

配置文件把“训练参数”和“代码逻辑”分开了。这样以后换模型、换数据、调超参数时，不需要直接改 Python 代码。

### 4. 数据是怎么流进训练的

原始数据先通过脚本处理：

- [scripts/prepare_sft_data.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/scripts/prepare_sft_data.py)

它的工作是：

1. 读取原始 JSONL
2. 规范成统一字段：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

3. 按比例切分 train / val
4. 写入 `data/processed/`

训练阶段由 [src/data/sft_dataset.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/sft_dataset.py) 负责把每一条样本变成模型输入。

核心逻辑：

1. 把 `instruction` 和 `input` 填进模板，生成 prompt
2. 把 `output` 作为 response 接到后面
3. tokenizer 编码
4. 构造 `input_ids`
5. 构造 `labels`

其中最关键的一点是：

- prompt 部分的 `labels` 被设成 `-100`
- response 部分的 `labels` 才保留真实 token id

这表示：

```text
模型训练时只学习“回答部分”，不对提示词部分计算 loss
```

关键代码位置：

- [src/data/sft_dataset.py:32](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/sft_dataset.py#L32)

我的理解：

这就是 `response-only loss`。它能让模型重点学习“怎么回答”，而不是浪费能力去拟合固定模板本身。

### 5. collator 在做什么

[src/data/collators.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/collators.py) 的作用是把不同长度的样本拼成一个 batch。

它会分别处理：

- `input_ids`
- `attention_mask`
- `labels`

具体来说：

- `input_ids` 用 `pad_token_id` 补齐
- `attention_mask` 用 `0` 补齐
- `labels` 用 `-100` 补齐

我的理解：

`collator` 解决的是“一个 batch 里的句子长度不一样怎么办”。如果没有它，模型没法稳定地按 batch 训练。

### 6. infer 推理时发生了什么

执行命令：

```bash
.venv/bin/python infer.py --model-path checkpoints/sft --prompt "解释什么是监督微调。"
```

内部流程：

1. [infer.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/infer.py) 读取参数
2. 加载 tokenizer
3. 加载模型
4. 根据当前机器判断用 `cpu` 还是 `cuda`
5. 用固定模板把用户输入包装成：

```text
### Instruction:
...

### Input:
...

### Response:
```

6. 调用 [src/eval/generate.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/eval/generate.py) 的 `generate_responses()`
7. 用 `model.generate()` 生成文本
8. 把前面的 prompt 部分裁掉，只保留新生成的回答

关键代码位置：

- [infer.py:21](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/infer.py#L21)
- [src/eval/generate.py:6](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/eval/generate.py#L6)

我的理解：

推理本质上和训练使用的是同一种 prompt 风格，只不过训练时模型看见“标准答案”，推理时模型需要自己续写答案。

### 7. 模型加载模块的作用

[src/models/policy_model.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/models/policy_model.py) 负责统一加载因果语言模型。

它现在做的事情比较少，主要是：

- 调 `AutoModelForCausalLM.from_pretrained(...)`
- 如果模型没有 `pad_token_id`，就把它对齐到 `eos_token_id`

我的理解：

这是一个“封装层”，价值在于以后如果我们要加：

- LoRA
- 量化
- device map
- 不同模型家族的兼容逻辑

就不需要把这些细节散落在每个训练脚本里。

### 8. 目前 PPO / GRPO / Distill 的状态

像下面这些入口：

- [train_ppo.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_ppo.py)
- [train_grpo.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_grpo.py)
- [train_distill.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/train_distill.py)

目前还没有真正开始训练，只是完成了：

1. 读取配置
2. 设置种子
3. 创建一个 scaffold
4. 打印后续需要实现的方向

例如 [src/trainers/ppo_trainer.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/trainers/ppo_trainer.py) 现在只是一个占位版本。

我的理解：

这很正常，因为项目是按阶段搭建的。现在我们已经先把最底层的 `Base -> SFT -> Infer` 打通了，后续才能继续往 reward 和 RL 对齐扩展。

### 9. 我现在对整个项目结构的理解

这个项目可以分成三层：

1. 入口层
   - `train_sft.py`
   - `infer.py`
   - `train_ppo.py`

2. 组装层
   - `src/trainers/*`

3. 底层能力层
   - `src/data/*`
   - `src/models/*`
   - `src/eval/*`
   - `src/utils/*`

这三层的关系是：

```text
入口脚本
-> 调 trainer
-> trainer 组织 model / data / collator / config
-> Hugging Face Trainer 或 generate 真正执行
```

### 10. 当前阶段最重要的理解

我现在应该优先吃透下面这几个点：

1. 一条样本是如何从 JSON 变成 `input_ids / labels / attention_mask` 的
2. 为什么 `labels` 的 prompt 部分要设为 `-100`
3. `collator` 为什么必须存在
4. `trainer.train()` 背后其实是在做什么
5. 推理时为什么要把 prompt 包装成和训练一致的模板

### 11. 当前已验证通过的命令

```bash
.venv/bin/python scripts/prepare_sft_data.py
.venv/bin/python train_sft.py --config configs/sft.yaml
.venv/bin/python infer.py --model-path distilgpt2 --prompt "解释什么是监督微调。"
.venv/bin/python infer.py --model-path checkpoints/sft --prompt "解释什么是监督微调。"
```

### 12. 当前阶段的注意点

- 目前使用的是 `distilgpt2`，它不适合中文，所以中文效果较差。
- 这不代表训练逻辑有问题，主要是基座模型不适合中文任务。
- 如果后续换成更适合中文的小模型，推理表现会更合理。

### 13. 下一步最值得继续学的内容

接下来可以继续补下面几个学习主题：

1. 手工拆一条样本，观察 `input_ids / labels` 到底长什么样
2. 讲清楚 Hugging Face `Trainer` 在训练时具体做了什么
3. 开始实现并理解 `Reward` 模块
4. 再进入 `PPO / GRPO` 的训练逻辑

### 14. 手工拆一条样本：它是怎么变成 `input_ids / labels / attention_mask` 的

这里用 [data/raw/demo_instructions.jsonl](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/data/raw/demo_instructions.jsonl) 里的第一条样本举例：

```json
{
  "instruction": "将下面的句子翻译成英文。",
  "input": "人工智能正在改变软件开发。",
  "output": "Artificial intelligence is changing software development."
}
```

#### 第一步：先拼 prompt

它会被模板 [configs/sft.yaml](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/configs/sft.yaml#L11) 包装成：

```text
### Instruction:
将下面的句子翻译成英文。

### Input:
人工智能正在改变软件开发。

### Response:
```

然后把真实答案接到后面：

```text
Artificial intelligence is changing software development.
```

所以在训练里，一条样本其实不是只看 `output`，而是看：

```text
prompt + response + eos
```

#### 第二步：tokenizer 编码

当前用 `distilgpt2` 编码时，我实际看到的长度是：

- `prompt_ids` 长度：`66`
- `response_ids` 长度：`8`
- `eos_token_id`：`50256`

也就是说：

```text
input_ids 总长度 = 66 + 8 + 1 = 75
```

这里很值得注意的一点是：

- 中文在 `distilgpt2` 里会被切成很多碎 token
- 英文句子 token 数通常更紧凑

这也是为什么当前中文任务效果不理想，除了模型能力本身，tokenizer 也并不适合中文。

#### 第三步：构造 `input_ids`

在 [src/data/sft_dataset.py:44](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/sft_dataset.py#L44) 里：

```text
input_ids = prompt_ids + response_ids + eos
```

可以把它理解成：

```text
模型真正看到的是“问题 + 答案”
```

也就是说，训练阶段模型输入并不是只有问题。
它会看到完整上下文，然后做“下一个 token 预测”。

#### 第四步：构造 `labels`

在 [src/data/sft_dataset.py:45](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/sft_dataset.py#L45) 里：

```text
labels = [-100] * len(prompt_ids) + response_ids + eos
```

代入这条样本后，可以理解成：

- 前 66 个位置：全部是 `-100`
- 接下来 8 个位置：是真实答案对应的 token id
- 最后 1 个位置：是 `50256`

所以：

```text
labels 总长度也等于 75
```

这里的关键意义是：

- `input_ids` 让模型看到完整上下文
- `labels` 决定哪些位置参与 loss 计算

而 `-100` 的含义是：

```text
这个位置不要参与 loss
```

所以训练时：

- prompt 部分不计入损失
- response 部分才计入损失

这正是 `response-only loss` 的核心。

#### 第五步：构造 `attention_mask`

在 [src/data/sft_dataset.py:49](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/sft_dataset.py#L49) 里，单条样本先被设成：

```text
attention_mask = [1] * len(input_ids)
```

也就是这条样本自己的所有真实 token 都先标成 `1`。

如果这一条样本长度是 `75`，那么它的单样本 `attention_mask` 就是：

```text
[1, 1, 1, ..., 1]
```

#### 第六步：batch 时为什么还需要 collator

单条样本没有问题，但 batch 训练时，不同样本长度往往不同。

这时 [src/data/collators.py](/Users/bahesplanck/mini-align-llm/Mini-Align-LLM/src/data/collators.py) 会把短样本补齐：

- `input_ids` 用 `pad_token_id` 补
- `attention_mask` 用 `0` 补
- `labels` 用 `-100` 补

例如：

- 样本 A 长度 `75`
- 样本 B 长度 `60`

那么样本 B 会被补到 `75`：

```text
input_ids:      真实 token + pad
attention_mask: 1...1 + 0...0
labels:         真实 labels + -100...-100
```

这样做的原因是：

- 模型 batch 输入必须是同样长度的张量
- padding 位置不应该参与注意力和 loss 计算

#### 第七步：我现在应该怎样理解这一整套设计

一条 SFT 样本的真正训练含义可以概括成一句话：

```text
让模型先看到“问题上下文”，然后只为“答案部分”负责
```

也可以记成下面这个公式：

```text
input_ids      = prompt + response + eos
labels         = mask(prompt) + response + eos
attention_mask = 1 for real tokens, 0 for padding
```

其中：

```text
mask(prompt) = -100 * prompt_length
```

#### 第八步：这部分知识为什么重要

如果没有真正理解这一层，后面会很容易在这些问题上混乱：

- 为什么模型输入里会包含答案
- 为什么训练时不是只喂问题
- 为什么 `labels` 和 `input_ids` 看起来长度一样
- 为什么 prompt 虽然输入了，但却不计算 loss
- 为什么 batch 训练一定要有 `collator`

我的理解：

这一部分是整个 `SFT` 的地基。如果把这一层吃透，后面再看 `PPO / GRPO / Distill` 时，就不会把“模型输入”和“优化目标”混在一起。
