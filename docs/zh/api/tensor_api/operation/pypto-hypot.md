# pypto.hypot

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

逐元素计算input和other的平方和的平方根（即直角三角形的斜边长）。计算公式如下：

$$
res_i = \sqrt{input_i^2 + other_i^2}
$$

## 函数原型

```python
hypot(input: Tensor, other: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1. input和other都为Tensor时，数据类型应该相同。
2. 对于BF16和FP16类型，内部计算可能会提升精度以避免中间溢出。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
# 示例：计算两组直角边的斜边
# 第一组: (3, 4) -> 5
# 第二组: (5, 12) -> 13
a = pypto.tensor([3.0, 5.0], pypto.DT_FP32)
b = pypto.tensor([4.0, 12.0], pypto.DT_FP32)
out = pypto.hypot(a, b)
```

结果示例如下：

```python
输入数据a:   [3.0, 5.0]
输入数据b:   [4.0, 12.0]
输出数据out: [5.0, 13.0]
```
