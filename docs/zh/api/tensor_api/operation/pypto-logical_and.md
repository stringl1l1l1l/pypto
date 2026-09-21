# pypto.logical\_and

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

对两个输入的Tensor进行逐元素逻辑与（AND）运算。运算规则：

- 如果输入的Tensor为bool则True and True -\> True，其余情况皆为False。
- 如果输入的Tensor数值，会自动转换成True/False，0为False，非0为True。

## 函数原型

```python
logical_and(input: Tensor, other: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_BOOL，DT_INT16，DT_INT32。<br>不支持空Tensor；Shape仅支持1-4维，支持输入Tensor的数据类型不同，支持多维度广播到相同形状。Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_BOOL，DT_INT16，DT_INT32。<br>不支持空Tensor；Shape仅支持1-4维，支持输入Tensor的数据类型不同，支持多维度广播到相同形状。Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型为DT\_BOOL，形状为广播后的形状。

## 约束说明

1. TileShape与input、other维度保持一致；
2. 由于存在临时内存使用，TileShape大小有额外约束，假设TileShape为[a,b,c,d]，那么a*b*c*d*sizeof(input) + a*b*c*d*sizeof(other) + a*b*c*d*sizeof(BOOL) + 1.1875KB < 可用内存上限。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：非广播场景，输入input shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

示例2：广播场景，输入input shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2], pypto.DT_BOOL)
y1 = pypto.tensor([2], pypto.DT_BOOL)
z1 = pypto.logical_and(x, y1)
# 支持广播
y2 = pypto.tensor([2,2], pypto.DT_BOOL)
z2 = pypto.logical_and(x, y2)
```

结果示例如下：

```python
输入数据x:  [True, False]
输入数据y1: [True, True]
输入数据y2: [[True, False], [False, True]]
输出数据z1: [True, False]
输出数据z2: [[True, False], [False, False]]
```
