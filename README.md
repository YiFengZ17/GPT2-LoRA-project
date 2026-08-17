# GPT-2 LoRA Sentiment Adaptation

这个项目以 CS224n A3 中手写的 GPT-2 Small 为基础，研究 LoRA（Low-Rank Adaptation，低秩适配）在情感分类任务中的参数效率和模型效果。

## 当前文件

```text
GPT2——LoRA project/
├── model.py              # GPT-2 核心网络、语言模型损失和生成
├── weight_converter.py   # Hugging Face GPT-2 权重格式转换
├── run_experiment.py     # 单次 Frozen / LoRA / Full 实验
├── run_suite.py          # 多随机种子与 LoRA rank 消融的一键实验套件
├── environment.yml       # 可复现 Conda 环境
├── requirements.txt      # Python 依赖版本范围
├── LoRA详细讲义.md        # LoRA 理论讲义
└── README.md              # 项目入口与模块说明
```

没有迁移 A3 的 TinyStories 训练脚本、作业 snapshot、pytest 辅助文件，以及仅用于调试的权重名称列表。这些内容不属于 LoRA 情感适配的核心运行路径。

## 模块关系

```text
token ids [B,T]
      │
      ▼
Token Embedding + Position Embedding
      │
      ▼
DecoderBlock × 12
      │
      ├── LayerNorm → CausalAttention → Residual
      │
      └── LayerNorm → MLP             → Residual
      │
      ▼
Final LayerNorm
      │
      ├── get_hidden_states() → hidden [B,T,768]
      │                         后续连接情感分类头
      │
      └── lm_head             → logits [B,T,50257]
                                用于语言建模和生成
```

### `ModelConfig`

保存模型结构超参数，例如模型维度、注意力头数、层数、最大上下文长度和词表大小。其他所有模块都通过同一个配置构造，避免维度不一致。

### `CausalAttention`

计算 Q、K、V 和多头缩放点积注意力。因果遮罩（causal mask）确保位置 `t` 只能看到自己和之前的 token。后续 LoRA 将主要添加到这里的 `W_q` 和 `W_v`。

### `MLP`

对每个 token 独立执行 `d_model → 4*d_model → d_model` 的非线性变换。注意力负责 token 之间的信息交换，MLP 负责对每个 token 的特征进行加工。

### `DecoderBlock`

把注意力、MLP、LayerNorm 和两条残差连接（residual connection）组装成一层 GPT-2。`Transformer.backbone` 会顺序执行 12 个 `DecoderBlock`。

### `Transformer`

负责完整数据流：嵌入、位置编码、逐层 DecoderBlock、最终 LayerNorm 和语言模型头。Token Embedding 与 LM Head 使用 GPT-2 的权重绑定（weight tying），共享同一个参数矩阵。`get_hidden_states()` 暴露最终隐藏状态，之后情感分类模型可取最后一个真实 token 的 `[768]` 向量，再映射到 5 个情感类别。

### `weight_converter`

Hugging Face GPT-2 使用一个合并的 `c_attn` 保存 Q/K/V，而且部分权重来自 `Conv1D`，矩阵方向与 `nn.Linear` 不同。转换器负责转置这些权重、拆分 Q/K/V，并映射到本项目的参数名称。

## 最小验证

在已有 `cs224n` 环境中运行：

```bash
conda activate cs224n
cd "/home/yifeng/projects/Summer2026/GPT2——LoRA project"

python - <<'PY'
import torch
from model import ModelConfig, Transformer

config = ModelConfig(
    d_model=24,
    n_heads=3,
    n_layers=2,
    context_length=16,
    vocab_size=100,
)

model = Transformer(config)
input_ids = torch.randint(0, 100, (2, 8))

hidden = model.get_hidden_states(input_ids)
logits = model(input_ids)
loss = model.get_loss_on_batch(input_ids)

print("hidden:", hidden.shape)
print("logits:", logits.shape)
print("loss:", loss.shape)
PY
```

预期输出：

```text
hidden: torch.Size([2, 8, 24])
logits: torch.Size([2, 8, 100])
loss: torch.Size([])
```

## 服务器安装

```bash
git clone https://github.com/YiFengZ17/GPT2-LoRA-project.git
cd GPT2-LoRA-project
conda env create -f environment.yml
conda activate gpt2-lora
python -m pytest -q
```

第一次运行会从 Hugging Face 下载 GPT-2 权重、tokenizer 和 `SetFit/sst5`
数据集，之后使用本地缓存。若服务器无法访问 Hugging Face，可在国内网络使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

不要在日本 VPN 下使用该国内镜像。

## 先做冒烟测试

冒烟测试（smoke test）只验证整条训练链路，不作为实验结论：

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --mode lora --rank 8 --debug --device cuda --precision fp16 \
  --output-dir runs/smoke
```

## 一次跑完整实验

主实验固定使用完整 SST-5、5 个 epoch、3 个随机种子，并比较：

1. Frozen GPT-2 + 分类头基线；
2. LoRA rank 2、4、8、16 消融；
3. Full fine-tuning 上界。

共 `3 × (1 + 4 + 1) = 18` 次独立运行：

先运行 `nvidia-smi`，确认 0、1、2、3 确实是导师分配且当前可用的 GPU。不要占用其他人的卡。四张卡均可用时：

```bash
python run_suite.py \
  --output-dir runs/main-study \
  --device cuda \
  --gpus 0 1 2 3 \
  --parallel 4 \
  --precision fp16 \
  --epochs 5 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --max-length 128 \
  --num-workers 2 \
  --seeds 13 42 2026 \
  --ranks 2 4 8 16 \
  2>&1 | tee runs/main-study/launcher.log
```

建议在 `tmux` 或 `screen` 中运行，防止本地 SSH 断线终止训练。重新执行同一条命令时，已经完成的 run 会跳过；存在 `latest.pt` 的未完成 run 会从最近 epoch 恢复。

调度器会先单进程准备并缓存数据、tokenizer 和 GPT-2 权重，然后动态地让每张空闲 GPU 领取下一个实验。四张卡的显存不会合并，每个实验始终只使用一张卡；这种方式不会引入 DDP 的通信开销，适合彼此独立的 18 次实验。FP16 自动混合精度会同时用于训练和评估，GradScaler 状态也保存在断点中。

上述高吞吐命令依靠固定 seed 和三次独立重复控制随机性，但不要求 CUDA 逐算子确定性。若需要逐次运行尽可能完全复现，可额外添加 `--deterministic`，代价是部分算子可能变慢。

如果 Full fine-tuning 在 11GB 卡上仍发生 OOM，把所有模式统一改为 `--batch-size 4 --gradient-accumulation-steps 2`，有效 batch size 仍为 8，实验仍可公平比较。

若服务器没有 `tmux`/`screen` 且你没有 sudo，可用系统自带的 `nohup`：

```bash
mkdir -p runs/main-study
nohup python -u run_suite.py \
  --output-dir runs/main-study \
  --device cuda \
  --gpus 0 1 2 3 \
  --parallel 4 \
  --precision fp16 \
  --epochs 5 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --max-length 128 \
  --num-workers 2 \
  --seeds 13 42 2026 \
  --ranks 2 4 8 16 \
  > runs/main-study/launcher.log 2>&1 &

echo $! > runs/main-study/launcher.pid
tail -f runs/main-study/launcher.log
```

每个 run 保存：

- `config.json`：命令参数、Git commit 和状态；
- `history.json`：逐 epoch 的 loss、accuracy、macro-F1、类别准确率、混淆矩阵和耗时；
- `latest.pt` / `best.pt`：断点与最佳验证集模型；
- `result.json`：测试集指标、可训练参数比例、总耗时、峰值显存和运行环境；
- `train.log`：完整标准输出与报错。

套件根目录保存 `summary.csv`（每次运行）和 `aggregate.csv`（各方法跨随机种子的均值与样本标准差）。这些记录共同回答项目目的：LoRA 相比 Frozen 能提升多少效果、相比 Full 使用多少可训练参数和显存，以及 rank 对效果/成本的影响。

正式写结论时以测试准确率 (test accuracy)、宏平均 F1 (macro-F1)、均值±标准差、可训练参数比例、峰值显存和训练时间为证据，不把单个随机种子的最好结果当作总体结论。
