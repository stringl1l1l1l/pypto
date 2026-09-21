# pypto.tanh

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

对输入Tensor的每个元素应用双曲正切函数（tanh），计算公式为：

$$
\tanh(input) = \frac{e^{input} - e^{-input}}{e^{input} + e^{-input}} = \frac{e^{2 \cdot input} - 1}{e^{2 \cdot input} + 1}
$$

该函数将输入映射到 \((-1, 1)\)区间，常用于神经网络激活函数。

## 函数原型

```python
tanh(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：DT_FP16、DT_FP32、DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入元素经tanh函数映射到 \((-1, 1)\)区间的结果。

## 约束说明

1. TileShape与input维度保持一致；
2. 由于存在临时内存使用，当输入数据类型为DT_FP32，TileShape大小有额外约束，假设TileShape为 [...,H,W]（最后两维为H和W），那么：
    `input_size + output_size + 2 * (W_align8) * H * sizeof(float) + (W_align8 / 8) * H + 32 bytes < UB`
    其中，`W_align8 = (W + 7) / 8 * 8`
    （FP32：input + output + 2个float temp tile + 1个compare mask tile + 32 bytes对齐）

    对于DT_FP16/DT_BF16输入，需要满足：
    `input_size + output_size + 4 * (W_align8) * H * sizeof(float) + (W_align8 / 8) * H + 32 bytes < UB`
    （FP16/BF16：input + output + 4个float temp tile + 1个compare mask tile + 32 bytes对齐）
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.tanh(x)
```

结果示例如下：

```python
输入数据x: [-3.0, -1.0, 0.0, 1.0, 3.0]
输出数据y: [-0.9951, -0.7616, 0.0000, 0.7616, 0.9951]
```

计算过程说明：

- tanh(-3.0) ≈ -0.9951，接近 -1
- tanh(-1.0) ≈ -0.7616
- tanh(0.0) = 0.0
- tanh(1.0) ≈ 0.7616
- tanh(3.0) ≈ 0.9951，接近1
