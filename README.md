# GPT-2 LoRA Sentiment Adaptation

这个项目以 CS224n A3 中手写的 GPT-2 Small 为基础，研究 LoRA（Low-Rank Adaptation，低秩适配）在情感分类任务中的参数效率和模型效果。

## 当前文件

```text
GPT2——LoRA project/
├── model.py              # GPT-2 核心网络、语言模型损失和生成
├── weight_converter.py   # Hugging Face GPT-2 权重格式转换
├── requirements.txt      # 最小运行依赖
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

## 下一步

下一阶段将在这个基础上依次加入：

1. `LoRALinear`
2. 对 `W_q`、`W_v` 的 LoRA 注入
3. padding attention mask
4. SST-5 情感分类头
5. Frozen / LoRA / Full Fine-tuning 对比实验
