# LoRA 详细讲义

## 0. 一句话理解

LoRA（Low-Rank Adaptation，低秩适配）的核心是：

> 冻结预训练模型原来的大权重矩阵，只训练一个很小的低秩修正矩阵。

它不是重新训练 GPT-2，而是在 GPT-2 的部分线性层旁边添加一条很小的“修正支路”。

---

## 1. 为什么需要 LoRA？

假设 GPT-2 预训练完成后，有一个线性层：

$$
y=Wx
$$

其中：

```text
x    输入向量
W    预训练权重
y    输出向量
```

如果要让 GPT-2 完成情感分类，最直接的方法是全参数微调（full fine-tuning）：

```text
预训练 GPT-2
     │
     ▼
在情感数据上反向传播
     │
     ▼
更新 GPT-2 的所有参数
```

但这有几个问题：

- 要计算和保存所有参数的梯度（gradient）
- AdamW 要为每个参数保存优化器状态（optimizer states）
- 每个任务都要保存一份完整 GPT-2
- 模型越大，训练和存储成本越高

GPT-2 Small 大约有 1.24 亿参数。如果我们有三个任务：

```text
GPT-2 情感分类模型
GPT-2 释义识别模型
GPT-2 文本生成模型
```

全参数微调通常意味着保存三份完整模型。

LoRA 的想法是：

```text
共享同一份冻结的 GPT-2
        │
        ├─ 情感 LoRA 参数
        ├─ 释义 LoRA 参数
        └─ 生成 LoRA 参数
```

每个任务只保存很小的 LoRA 参数。

---

## 2. LoRA 的直觉

把预训练模型想象成一本已经出版的教科书。

全参数微调相当于：

> 为了适配一个新任务，把整本书重新抄写并修改一遍。

LoRA 相当于：

> 原书保持不变，只附上一份很薄的修改说明。

数学上，预训练参数是 $W$，任务需要的修改是：

$$
W'=W+\Delta W
$$

全参数微调直接学习一个完整的 $\Delta W$。

LoRA 假设：

> 下游任务需要的参数变化，可能主要集中在一个低维子空间（low-dimensional subspace）中。

因此，不直接学习完整的 $\Delta W$，而是把它写成两个小矩阵的乘积：

$$
\Delta W=BA
$$

所以：

$$
W'=W+BA
$$

其中 $BA$ 就是 LoRA 学到的任务修正。

这个方法由 Hu 等人在 [LoRA 论文](https://arxiv.org/abs/2106.09685)中提出。

---

## 3. 什么是“低秩”？

### 3.1 从一个简单矩阵开始

假设原来的线性层是：

$$
W\in\mathbb{R}^{4\times4}
$$

完整矩阵有 16 个参数：

$$
W=
\begin{bmatrix}
w_{11}&w_{12}&w_{13}&w_{14}\\
w_{21}&w_{22}&w_{23}&w_{24}\\
w_{31}&w_{32}&w_{33}&w_{34}\\
w_{41}&w_{42}&w_{43}&w_{44}
\end{bmatrix}
$$

如果完整学习一个 $\Delta W$，也要训练 16 个参数。

LoRA 可以选择秩：

$$
r=1
$$

然后定义：

$$
A\in\mathbb{R}^{1\times4},\qquad
B\in\mathbb{R}^{4\times1}
$$

例如：

$$
A=
\begin{bmatrix}
a_1&a_2&a_3&a_4
\end{bmatrix}
$$

$$
B=
\begin{bmatrix}
b_1\\b_2\\b_3\\b_4
\end{bmatrix}
$$

两者相乘：

$$
BA=
\begin{bmatrix}
b_1a_1&b_1a_2&b_1a_3&b_1a_4\\
b_2a_1&b_2a_2&b_2a_3&b_2a_4\\
b_3a_1&b_3a_2&b_3a_3&b_3a_4\\
b_4a_1&b_4a_2&b_4a_3&b_4a_4
\end{bmatrix}
$$

现在只需要：

```text
A：4 个参数
B：4 个参数
总计：8 个参数
```

而不是完整 $\Delta W$ 的 16 个参数。在真实模型中，矩阵通常是 `768 × 768`，节省比例会更加明显。

### 3.2 rank 的含义

秩（rank）可以理解为：

> 这个参数更新可以包含多少种独立的变化方向。

当 `r=1` 时：

$$
\Delta W=b_1a_1^T
$$

只有一个变化方向。

当 `r=2` 时：

$$
\Delta W=b_1a_1^T+b_2a_2^T
$$

一般来说：

$$
BA=\sum_{i=1}^{r}b_i a_i^T
$$

所以：

- `r` 小：参数少，但表达能力弱
- `r` 大：参数多，但表达能力强
- `r` 达到矩阵完整维度：逐渐接近完整矩阵更新

一个重要的数学性质是：

$$
\operatorname{rank}(BA)\le r
$$

因此叫作低秩适配（Low-Rank Adaptation）。

注意：不是说最终权重 $W+BA$ 的秩只有 $r$，而是说任务产生的更新 $\Delta W=BA$ 的秩最多为 $r$。

---

## 4. 完整的 LoRA 公式

对于普通线性层：

$$
y=Wx
$$

LoRA 将它改为：

$$
y=Wx+\frac{\alpha}{r}BAx
$$

| 符号 | 含义 |
|---|---|
| $W$ | 冻结的预训练权重 |
| $A$ | 第一个可训练低秩矩阵 |
| $B$ | 第二个可训练低秩矩阵 |
| $r$ | LoRA 的秩 |
| $\alpha$ | LoRA 缩放系数 |
| $\alpha/r$ | 最终缩放比例 |

数据流如下：

```text
                         ┌──────────────┐
                    ┌───▶│ 冻结的 W(x)  │───┐
                    │    └──────────────┘   │
x [..., d_in] ──────┤                       ├──▶ 相加 ──▶ y [..., d_out]
                    │    ┌──────────────┐   │
                    └───▶│ A: d_in → r  │   │
                         └──────┬───────┘   │
                                ▼            │
                         [..., r]            │
                                │            │
                         ┌──────▼───────┐   │
                         │ B: r → d_out │───┘
                         └──────────────┘
                                │
                         乘以 α/r
```

---

## 5. Tensor 形状

PyTorch 的 `nn.Linear(d_in, d_out)` 内部权重形状是：

```text
weight [d_out, d_in]
```

设：

```text
x     [B,T,d_in]
W     [d_out,d_in]
A     [r,d_in]
B     [d_out,r]
```

则完整过程是：

```text
x
[B,T,d_in]
│
├─ base linear
│  x @ Wᵀ
│  └─ [B,T,d_out]
│
└─ LoRA branch
   x @ Aᵀ
   └─ [B,T,r]
      │
      @ Bᵀ
      └─ [B,T,d_out]
         │
         × alpha/r

相加
└─ [B,T,d_out]
```

论文公式一般把 $x$ 写成列向量：

$$
Wx+BAx
$$

而 PyTorch 经常把 batch 维放在前面，因此实际计算看起来像：

$$
xW^T+xA^TB^T
$$

两种写法表达的是同一件事，只是向量摆放方向不同。

---

## 6. GPT-2 中的具体例子

GPT-2 Small 的模型维度：

$$
d_{\text{model}}=768
$$

注意力投影为：

```text
W_q: 768 → 768
W_k: 768 → 768
W_v: 768 → 768
W_o: 768 → 768
```

假设给 `W_q` 添加 `r=8` 的 LoRA。

### 原始矩阵

$$
W_q\in\mathbb{R}^{768\times768}
$$

参数量：

$$
768\times768=589,824
$$

### LoRA 矩阵

$$
A_q\in\mathbb{R}^{8\times768},\qquad
B_q\in\mathbb{R}^{768\times8}
$$

参数量：

$$
8\times768+768\times8=12,288
$$

对比：

```text
完整 W_q 更新：589,824 个参数
LoRA r=8：     12,288 个参数
```

单个投影约缩小：

$$
\frac{589,824}{12,288}=48
$$

也就是约 48 倍。

---

## 7. 本项目中有多少 LoRA 参数？

计划在 GPT-2 的每层修改：

```text
W_q
W_v
```

GPT-2 Small 有 12 层。

一个投影的 LoRA 参数：

$$
r(768+768)=1536r
$$

每层两个投影：

$$
2\times1536r=3072r
$$

12 层：

$$
12\times3072r=36864r
$$

| rank | Q/V LoRA 参数 | 加上 SST-5 分类头 | 约占 GPT-2 |
|---:|---:|---:|---:|
| 4 | 147,456 | 151,301 | 0.12% |
| 8 | 294,912 | 298,757 | 0.24% |
| 16 | 589,824 | 593,669 | 0.48% |

SST-5 分类头的参数是：

$$
768\times5+5=3845
$$

这就是为什么 LoRA 被称为参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）。

---

## 8. 为什么 A 随机初始化，而 B 初始化为零？

通常使用：

```text
A：随机初始化
B：全零初始化
```

训练刚开始时：

$$
BA=0
$$

所以：

$$
y=Wx+\frac{\alpha}{r}\cdot0=Wx
$$

这意味着加入 LoRA 后，模型一开始与原预训练模型完全相同。

如果 $A$ 和 $B$ 都随机初始化，那么一加入 LoRA，模型输出就会突然发生随机变化。

### 如果 B=0，模型还能学习吗？

可以。

第一步反向传播时：

- $B$ 可以收到非零梯度并开始更新
- 因为 $B=0$，此时 $A$ 的梯度可能是零
- 更新一次之后 $B\neq0$
- 后续步骤中 $A$ 也会获得梯度

所以训练不会被永远卡住。

第一个单元测试应该是：

```python
output_before = original_linear(x)
output_after = lora_linear(x)

assert torch.allclose(output_before, output_after)
```

---

## 9. alpha 是什么？

LoRA 完整公式：

$$
y=Wx+\frac{\alpha}{r}BAx
$$

其中：

$$
\text{scaling}=\frac{\alpha}{r}
$$

`r` 增加时，低秩支路的表达能力和参数量都增加。除以 `r` 可以避免 rank 改变时输出规模发生过大的变化。

例如：

```text
r = 8
alpha = 16
scaling = 16 / 8 = 2
```

或者：

```text
r = 16
alpha = 32
scaling = 32 / 16 = 2
```

常见经验是：

```text
alpha = r
或者
alpha = 2r
```

但它是超参数，不是数学上必须如此。本项目可以固定 `alpha = 2r`，让实验只改变 `r`。

---

## 10. LoRA dropout 是什么？

LoRA 经常只在低秩支路上使用 dropout：

$$
y=Wx+\frac{\alpha}{r}BA(\operatorname{Dropout}(x))
$$

```text
                    ┌──▶ W(x) ───────────────┐
x ──────────────────┤                         ├─▶ 相加
                    └──▶ Dropout → A → B ────┘
```

原始模型支路不会被这个 dropout 修改。

它的作用是减少 LoRA 参数对训练集的过拟合（overfitting）。对于 SST-5 这样较小的数据集，可以从以下配置开始：

```text
lora_dropout = 0.05
```

评估时调用 `model.eval()`，dropout 就会关闭。

---

## 11. 为什么经常把 LoRA 加到 Q 和 V？

注意力计算为：

$$
Q=XW_q,\quad K=XW_k,\quad V=XW_v
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}
\right)V
$$

直觉上：

- $Q$：当前 token 想寻找什么信息
- $K$：每个 token 提供什么匹配标签
- $V$：真正被读取和传递的信息
- $W_o$：把多个注意力头的结果重新组合

修改 $W_q$ 可以改变模型在情感分类时关注什么；修改 $W_v$ 可以改变被关注的信息怎样传给后续网络。

例如：

> The movie looked promising, but it was painfully boring.

为了判断情感，模型可能需要增强：

- 对转折词 `but` 后面内容的注意
- 对 `painfully boring` 的负面表达
- 对前半句正向词 `promising` 的抑制

LoRA 也可以加在 $W_k$、$W_o$、MLP 或所有线性层上。但加得越多，参数和实验变量越多。因此一周项目先使用 `Q + V`。

---

## 12. LoRA 如何完成情感分类？

LoRA 本身不直接输出情感类别。

原 GPT-2 输出的是词表 logits：

```text
[B,T,50257]
```

因为语言模型头（LM head）用于预测下一个 token。情感分类需要另一个分类头：

```text
[B,768] → Linear(768,5) → [B,5]
```

完整流程：

```text
input_ids
[B,T]
│
▼
GPT-2 + LoRA
[B,T,768]
│
▼
取最后一个真实 token 的 hidden state
[B,768]
│
▼
分类头
[B,5]
│
▼
CrossEntropyLoss(logits, labels)
[]
```

训练时通常更新：

```text
LoRA A/B        requires_grad=True
分类头           requires_grad=True

GPT-2 原参数      requires_grad=False
Token embedding  requires_grad=False
LM head          通常不使用或冻结
```

### 为什么取最后一个 token？

GPT-2 是因果模型（causal model）。最后一个真实 token 的隐藏状态已经通过注意力看到了它前面的整句话，因此可以作为句子表示（sentence representation）。

如果 batch 中有 padding，不能简单写：

```python
hidden[:, -1, :]
```

因为最后一个位置可能是 padding。应通过 `attention_mask` 找到每个句子的最后一个真实 token。

---

## 13. LoRA 的训练过程

```text
1. 加载预训练 GPT-2
          │
          ▼
2. 冻结 GPT-2 所有原始参数
          │
          ▼
3. 给每层 W_q/W_v 添加 LoRA A/B
          │
          ▼
4. 添加情感分类头
          │
          ▼
5. 前向传播
          │
          ▼
6. 计算 CrossEntropyLoss
          │
          ▼
7. 反向传播
          │
          ▼
8. 只更新 A、B 和分类头
```

伪代码：

```python
model = load_pretrained_gpt2()

for parameter in model.parameters():
    parameter.requires_grad = False

add_lora_to_q_and_v(model, rank=8)
model.classifier = nn.Linear(768, 5)

optimizer = AdamW(
    parameters_that_require_grad,
    lr=2e-4,
)
```

反向传播时：

```text
原始 W_q.grad           None
原始 W_v.grad           None
LoRA A.grad             非零
LoRA B.grad             非零
classifier.weight.grad  非零
```

由于零初始化，第一次反向传播时 `lora_A` 的梯度可能为零，这是正常现象。

---

## 14. LoRA 节省了什么？

### 14.1 节省可训练参数

```text
Full fine-tuning：约 124M
LoRA r=8：       约 0.299M
```

### 14.2 节省优化器状态

AdamW 通常为每个可训练参数保存参数、梯度、一阶动量和二阶动量。冻结原参数后，不需要为它们保存完整的梯度和优化器状态。

### 14.3 节省 checkpoint 空间

全参数微调要保存整个模型。LoRA 只需要保存：

```text
所有 LoRA A
所有 LoRA B
分类头
LoRA 配置
```

基础 GPT-2 可以单独共享。

### 14.4 不会等比例节省计算

LoRA 仍然需要运行完整 GPT-2 前向传播、计算各层激活，并让梯度经过网络传到 LoRA 参数。因此：

```text
可训练参数下降 99%+
```

不意味着：

```text
训练时间下降 99%+
```

LoRA 的主要优势通常是参数、优化器内存、显存和存储效率，而不是让完整 Transformer 计算消失。

---

## 15. LoRA 能合并回原权重吗？

可以。

训练结束后：

$$
W_{\text{merged}}
=
W+\frac{\alpha}{r}BA
$$

之后推理直接使用：

$$
y=W_{\text{merged}}x
$$

合并前：

```text
x ──▶ W ──────────┐
                  ├─▶ y
x ──▶ A ──▶ B ────┘
```

合并后：

```text
x ──▶ W_merged ──▶ y
```

一个很好的测试是：

```python
output_unmerged = model(x)

model.merge_lora_weights()

output_merged = model(x)

assert torch.allclose(
    output_unmerged,
    output_merged,
    atol=1e-5,
)
```

---

## 16. LoRA 与其他方法的区别

| 方法 | 做法 |
|---|---|
| 全参数微调（full fine-tuning） | 更新所有模型参数 |
| 线性探测（linear probing） | 冻结模型，只训练分类头 |
| LoRA | 冻结模型，训练低秩权重更新和分类头 |
| Adapter | 在 Transformer 中插入新的小型网络层 |
| Prompt tuning | 学习若干连续的虚拟 prompt 向量 |
| 量化（quantization） | 用更少的数值位数表示权重 |
| QLoRA | 量化冻结的基础模型，同时训练 LoRA |

LoRA 不是量化，也不是模型剪枝（pruning）。它的本质是参数高效模型适配（parameter-efficient model adaptation）。

---

## 17. 为什么低秩更新可能有效？

预训练 GPT-2 已经学习了大量语言知识：

- 词语含义
- 句法结构
- 长距离关系
- 正面与负面表达
- 常见语义组合

情感分类并不要求模型重新学习整个英语，只需要让模型：

- 更关注与情感有关的方向
- 重新组合已有语言特征
- 把句子表示映射到情感类别

因此，任务需要的权重变化可能不需要覆盖整个高维参数空间。

可以想象模型有 768 个特征方向，但情感分类主要依赖：

```text
正面程度
负面程度
否定关系
转折结构
情绪强度
反讽线索
```

实际所需的适配方向可能远少于 768。这就是低秩假设（low-rank hypothesis）。

但它不是保证：

- `r` 太小可能欠拟合（underfitting）
- 任务越复杂，可能需要更大的 `r`
- 不同目标层可能需要不同 rank
- LoRA 不一定总能匹配全参数微调

这些问题需要通过实验回答。

---

## 18. rank 太小或太大会怎样？

### rank 太小

例如 `r=1`：参数极少，但修改方向太有限，可能出现训练损失下降缓慢、准确率低和难以适应复杂情感表达。

### rank 合适

例如 `r=8`：参数少，但表达能力可能已经足够。很多任务可能在较小 rank 就达到性能饱和。

### rank 太大

例如 `r=128`：参数更多、训练更慢、更容易过拟合，低秩优势也会减弱。

rank 增加不保证准确率单调上升。例如实验可能出现：

```text
r=4     49.2%
r=8     51.7%
r=16    51.9%
r=32    51.5%
```

这些数字只是说明性的假设，不是已经验证的实验结果。真实结果必须通过本项目实验得到。

---

## 19. 手写 LoRA 的基本结构

下面是结构示意，不是最终项目实现：

```python
class LoRALinear(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()

        self.base = base_linear
        self.lora_A = nn.Linear(
            base_linear.in_features,
            rank,
            bias=False,
        )
        self.lora_B = nn.Linear(
            rank,
            base_linear.out_features,
            bias=False,
        )

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        initialize_A_randomly()
        initialize_B_to_zero()

    def forward(self, x):
        base_output = self.base(x)

        lora_output = self.lora_B(
            self.lora_A(
                self.dropout(x)
            )
        )

        return base_output + self.scaling * lora_output
```

`base_linear` 必须直接使用原来加载好的 GPT-2 权重，不能新建一个随机 Linear 替代它。

---

## 20. 实现时必须验证的五件事

### 测试一：初始输出一致

因为 `B=0`：

```python
original_output = original_linear(x)
adapted_output = lora_linear(x)

assert torch.allclose(original_output, adapted_output)
```

### 测试二：原参数被冻结

```python
assert lora_layer.base.weight.requires_grad is False
```

### 测试三：只有目标参数可训练

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name, parameter.numel())
```

输出中应该主要只有：

```text
lora_A.weight
lora_B.weight
classifier.weight
classifier.bias
```

### 测试四：梯度正确

运行一次反向传播后：

```python
assert lora_layer.base.weight.grad is None
assert lora_layer.lora_B.weight.grad is not None
assert classifier.weight.grad is not None
```

由于零初始化，第一次反向传播时 `lora_A` 的梯度可能为零，这是正常现象。

### 测试五：参数量正确

对于 `d_in=d_out=768, r=8`：

```text
A：8 × 768   = 6,144
B：768 × 8   = 6,144
合计          = 12,288
```

如果不是这个数量，通常说明矩阵方向写反或错误加入了 bias。

---

## 21. 一个完整的形状例子

假设：

```text
B = 2
T = 32
d_model = 768
r = 8
```

对于 `W_q`：

```text
hidden states
[2,32,768]
│
├─ 原始 W_q
│  └─ [2,32,768]
│
└─ LoRA
   │
   ├─ dropout
   │  └─ [2,32,768]
   │
   ├─ A: 768 → 8
   │  └─ [2,32,8]
   │
   ├─ B: 8 → 768
   │  └─ [2,32,768]
   │
   └─ × alpha/r
      └─ [2,32,768]

相加
└─ q [2,32,768]

reshape heads
└─ [2,32,12,64]

transpose
└─ [2,12,32,64]
```

后面的多头注意力过程完全不变。因此 LoRA 不改变模型的外部输入输出形状，只改变线性层内部如何计算结果。

---

## 22. 最容易犯的错误

### 错误一：没有冻结原始参数

如果 $W$ 也在更新，就不是纯 LoRA 微调。

### 错误二：把 W 替换成 BA

LoRA 是：

$$
Wx+BAx
$$

不是：

$$
BAx
$$

原来的预训练能力必须保留。

### 错误三：A、B 都初始化为零

如果二者全为零，$B$ 的梯度依赖 $A$，而 $A$ 的梯度依赖 $B$，两者可能都无法开始学习。所以通常是 A 随机、B 为零。

### 错误四：认为 LoRA 参数少，所以训练计算也少 99%

完整 Transformer 前向传播仍然存在。

### 错误五：使用最后一个 padding 的隐藏状态分类

必须找最后一个真实 token，或者构造专门的分类 token。

### 错误六：忘记训练分类头

LoRA 只适配 GPT-2 内部表示，分类头仍然需要训练。

### 错误七：比较不同 rank 时同时乱改其他超参数

为了让实验结论可信，应固定：

```text
目标层
训练集
batch size
训练 epoch
dropout
随机种子
alpha/r 比例
```

主要只改变 `rank`。

---

## 23. 本项目真正要回答的问题

这个项目不只是证明“LoRA 能运行”，而是研究：

$$
\text{效果}\quad\text{vs.}\quad\text{参数效率}
$$

实验要回答：

1. 只训练分类头的效果是多少？
2. 加入 LoRA 后提高多少？
3. `r=4、8、16` 有什么差别？
4. LoRA 与全参数微调差多少？
5. LoRA 节省多少可训练参数和 checkpoint 空间？
6. 参数节省是否真的转化为训练速度和显存优势？
7. LoRA 仍然容易在哪些情感句子上出错？

最终最理想的结论不是：

> `r=16` 的准确率最高。

而是类似：

> `r=8` 只训练约 0.24% 的 GPT-2 参数，已经达到全参数微调 98% 的相对性能；继续增加到 `r=16` 的收益很小，说明该任务的有效参数更新可能具有较低的内在维度（intrinsic dimension）。

这句话中的具体数字必须由实验得到，不能提前假设。

---

## 24. 核心总结

```text
预训练权重
W [d_out,d_in]
冻结
      │
      ├─────────────────▶ W(x)
      │
输入 x│
      └─▶ A [r,d_in]
           │
           ▼
          [...,r]
           │
           ▼
         B [d_out,r]
           │
           ▼
       (alpha/r)BAx
           │
           ▼
与 W(x) 相加
           │
           ▼
最终输出
```

需要记住六点：

1. LoRA 冻结预训练权重 $W$。
2. LoRA 学习低秩更新 $\Delta W=BA$。
3. `r` 控制参数量和适配能力。
4. 通常 A 随机初始化、B 初始化为零。
5. 训练时只更新 A、B 和分类头。
6. 对情感分类，最终仍需 `768 → 5` 的分类头。

## 理解检查

请尝试用自己的话回答：

> 如果 LoRA 初始化时 $B=0$，为什么模型的初始输出不会改变，但训练仍然能够开始？

