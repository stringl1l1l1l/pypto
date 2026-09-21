# pypto.isfinite

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

判断张量中的元素值是否为有限值。

当张量类型为整型时，返回一个全为True的布尔类型、与输入张量形状大小一致的张量。

当张量类型为浮点数类型时，仅inf/nan/-inf不是有限值，结果中对应元素位置的值为False，其余为True。

## 函数原型

```python
isfinite(self: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32，DT_UINT8，DT_INT8，DT_UINT16，DT_INT16，DT_UINT32，DT_INT32，DT_UINT64，DT_INT64。<br>不支持空Tensor；形状大小支持1-4维；形状大小中对应元素的个数不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出张量，张量的数据类型为布尔类型DT_BOOL，形状大小与输入张量的形状大小一致。

## 约束说明

1. 仅支持DT_FP16，DT_BF16，DT_FP32，DT_UINT8，DT_INT8，DT_UINT16，DT_INT16，DT_UINT32，DT_INT32，DT_UINT64，DT_INT64等数据类型。
2. TileShape以及ViewShape的尾轴必须按照输出张量的类型32Byte对齐，由于输出张量为布尔类型，因此TileShape以及ViewShape的尾轴必须是32的倍数。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 32)
```

### 接口调用示例

```python
self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
out = pypto.isfinite(self)
```

结果示例如下：

```python
输入数据self: [[1 nan 3],
               [inf 1 1],
               [1 1 -inf]]
输出数据out: [[True False True],
             [False True True],
             [True True False]]
```
