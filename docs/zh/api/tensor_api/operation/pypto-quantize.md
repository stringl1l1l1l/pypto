# pypto.quantize

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

将高精度浮点数据转换为低精度格式，当前支持

- 输入DT_FP32的Tensor通过对称量化转换为DT_INT8的Tensor
  $$
  \text{dst} = round(\text{input} * \text{scale})
  $$
- 输入DT_FP32的Tensor通过非对称量化转换为DT_UINT8的Tensor
  $$
  \text{dst} = round(\text{input} * \text{scale} + \text{zero\_points})
  $$

## 函数原型

```python
quantize(input: Tensor, scale: Tensor, otype: DataType, axis: int, zero_points: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。<br>shape记为 [..., row, col] |
| scale  | 输入      | 缩放因子。<br>支持的类型为：Tensor。<br>Tensor数据类型与input一致，支持：DT_FP32;<br>不支持空Tensor；<br>Shape比input少一维，仅支持1-3维；<br>Shape Size不大于2147483647（即INT32_MAX）;<br>axis = -1或input.shape.size() -1时，shape = [..., row]<br>axis = -2或input.shape.size() -2时，shape = [..., col]|
| otype  | 输入      | 返回值的数值类型<br>目前支持int8和uint8，分别对应对称量化和非对称量化。|
| axis  | 输入      | 指定量化压缩的轴<br>目前支持末尾两轴，即 -1/-2或者input.shape.size() -1/input.shape.size()-2<br>**当input为1D时，仅支持-1** |
| zero_points  | 输入      | 可选的非对称量化的偏移因子<br>支持的类型为：Tensor。<br>Tensor数据类型与input一致，支持：DT_FP32;<br>支持空Tensor；<br>Shape比input少一维，仅支持1-3维；<br>Shape Size不大于2147483647（即INT32_MAX）;<br>axis = -1或input.shape.size() -1时，shape = [..., row]<br>axis = -2或input.shape.size() -2时，shape = [..., col] |

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
x = pypto.tensor([3, 4], pypto.DT_FP32)
scale = pypto.tensor([3], pypto.DT_FP32)
zero_points = pypto.tensor([3], pypto.DT_FP32)

# fp32 -> int8对称量化
y1 = pypto.quantize(x, scale, pypto.DT_INT8, -1, None)
# fp32 -> uint8非对称量化
y2 = pypto.quantize(x, scale, pypto.DT_UINT8, -1, zero_points)
```

结果示例如下：

```python
Input  x:[[1.1, -2.2, 3.3, -4.4], [1.1, -2.2, 3.3, -4.4], [1.1, -2.2, 3.3, -4.4]]
Input  scale:[1.0, 1.0, 1.0]
Input zero_points:[-5.0, -5.0, -5.0]
Output y1:[[1, -2, 3, -4], [1, -2, 3, -4], [1, -2, 3, -4]]
Output y2:[[6, 3, 8, 1], [6, 3, 8, 1], [6, 3, 8, 1]]
```
