# pypto.dequantize

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

将量化后的低精度数据转换为高精度格式，并应用缩放(scale)和偏移(zero_points)参数，当前支持

- 输入DT_INT8/DT_INT16的Tensor反量化为DT_FP32的Tensor
  $$
  \text{dst} = ([float]\text{input} + \text{zero\_points}) * \text{scale}
  $$

## 函数原型

```python
dequantize(input: Tensor, scale: Tensor, otype: DataType, axis: int, zero_points: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT8/DT_INT16;<br>不支持空Tensor；<br>Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。<br>shape记为 [..., row, col] |
| scale  | 输入      | 缩放因子。<br>支持的类型为：Tensor。<br>Tensor数据类型与otype一致，支持：DT_FP32；<br>不支持空Tensor；<br>Shape比input少一位维，仅支持1-3维；<br>Shape Size不大于2147483647（即INT32_MAX）；<br>axis = -1或input.shape.size() -1时，shape = [..., row]<br>axis = -2或input.shape.size() -2时，shape = [..., col]|
| otype  | 输入      | 返回值的数值类型<br>目前支持DT_FP32。|
| axis  | 输入      | 指定反量化压缩的轴<br>目前支持末尾两轴，即 -1/-2或者input.shape.size() -1/input.shape.size()-2<br>**当input为1D时，仅支持-1** |
| zero_points  | 输入      | 可选的非对称量化的偏移因子<br>支持的类型为：Tensor。<br>Tensor数据类型与otype一致，支持：DT_FP32；<br>支持空Tensor；<br>Shape比input少一位维，仅支持1-3维；<br>Shape Size不大于2147483647（即INT32_MAX）；<br>axis = -1或input.shape.size() -1时，shape = [..., row]<br>axis = -2或input.shape.size() -2时，shape = [..., col]|

## 返回值说明

返回输出Tensor，Tensor的数据类型由otype指定，Shape与input相同。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
x = pypto.tensor([3, 4], pypto.DT_INT8)
scale = pypto.tensor([3, 1], pypto.DT_FP32)
zero_points = pypto.tensor([3, 1], pypto.DT_FP32)

# fp32 -> int8对称反量化
y1 = pypto.dequantize(x, scale, pypto.DT_FP32, -1, None)
# fp32 -> uint8非对称反量化
y2 = pypto.dequantize(x, scale, pypto.DT_FP32, -1, zero_points)
```

结果示例如下：

```python
Input  x:[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
Input  scale:[1.0, 1.0, 1.0]
Input  zero_points:[-2.0, -2.0, -2.0]
Output y1:[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
Output y2:[[-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0]]
```
