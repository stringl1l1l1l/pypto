# pypto.relu

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

对input的每个元素进行整流线性单元（Rectified Linear Unit）运算，即只保留正数部分，负数变为0。计算公式如下：

$$
res_i = \max(0, input_i)
$$

## 函数原型

```python
relu(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和Shape与input相同。

## 约束说明

1. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT16，DT_INT32，DT_INT64，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT16，DT_INT32，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT16，DT_INT32，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id6 -->
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

### 接口调用示例

```python
input_tensor = pypto.tensor([[-2.0, 0.0, 3.0]], pypto.DT_FP32)
out = pypto.relu(input_tensor)
```

结果示例如下：

```python
输入数据input：[[-2.0 0.0 3.0]]
输出数据out：[[ 0.0 0.0 3.0]]
```
