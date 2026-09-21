# pypto.sign

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

计算输入Tensor中每个元素的符号值，逐元素运算。计算公式如下：

$$
\text{sign}(x) = \begin{cases}
-1 & \text{if } x < 0 \\
0 & \text{if } x = 0 \\
1 & \text{if } x > 0
\end{cases}
$$

## 函数原型

```python
sign(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape与输入Tensor一致，数据类型与输入Tensor一致，其元素为输入Tensor对应元素的符号值（-1、0或1）。

## 约束说明

1. TileShape与input维度保持一致；
2. 由于存在临时内存使用，设置TileShape时需保证输入Tile、输出Tile及临时空间的总占用小于可用UB。假设TileShape的最后两维为`H`和`W`（一维TileShape取`H = 1`），各数据类型对应的临时空间如下：

   | 输入数据类型 | 实际工作数据类型 | 尾轴对齐长度 | 临时空间大小（字节） |
   |--------------|------------------|--------------|----------------------|
   | `DT_INT8` | `DT_FP16` | `W_align = CeilAlign(W, 16)` | `H * W_align * sizeof(DT_FP16)` |
   | `DT_FP16` | `DT_FP16` | `W_align = CeilAlign(W, 16)` | `2 * H * W_align * sizeof(DT_FP16) + 32` |
   | `DT_BF16` | `DT_FP32` | `W_align = CeilAlign(W, 8)` | `2 * H * W_align * sizeof(DT_FP32) + 32` |
   | `DT_FP32` | `DT_FP32` | `W_align = CeilAlign(W, 8)` | `2 * H * W_align * sizeof(DT_FP32) + 32` |
   | `DT_INT16`、`DT_INT32`、`DT_INT64` | 与输入一致 | 不涉及 | `32` |

   浮点计算路径中的两块等大临时空间分别用于工作数据和比较掩码，额外的32字节用于标量临时块。`DT_BF16`输入在进入Sign TileOp前会通过AutoCast转换为`DT_FP32`，因此按`DT_FP32`路径申请临时空间；计算完成后，结果再转换回`DT_BF16`。
3. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_INT16，DT_INT32，DT_INT64。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_INT16，DT_INT32。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_INT16，DT_INT32。
   <!-- end id6 -->
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sign(x)
```

结果示例如下：

```python
输入数据x: [-5.0, 0.0, 5.0, 10.0, -2.0]
输出数据y: [-1.0, 0.0, 1.0, 1.0, -1.0]
```
