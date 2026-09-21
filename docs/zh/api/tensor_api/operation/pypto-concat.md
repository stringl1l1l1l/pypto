# pypto.concat

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

将输入的多个Tensor沿指定维度（dim）拼接，返回一个拼接后的Tensor。

## 函数原型

```python
concat(tensors: List[Tensor], dim: int = 0) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                                                                                                                                                     |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tensors | 输入      | 源操作数。支持的类型为：Tensor。Tensor支持的数据类型为：DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_FP16，DT_FP32，DT_BF16，DT_INT64，DT_UINT64。不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定进行拼接的维度。支持的数据类型为：int，默认为0。                                                                                                                                                      |

## 返回值说明

返回输出Tensor，Tensor的数据类型和tensors的任一tensor数据类型相同，Shape与tensors任一tensor相同（除dim对应维度），dim对应维度为tensors各个tensor对应维度之和。

## 约束说明

1.源操作数tensors的大小需要大于等于2，即len\(tensors \)\>=2；小于等于128。（支持输入一个tensor情况，精度暂时不保证）；

2.输入tensor数据类型相同、维度数量相同，并且除待拼接维度（dim）之外的每个维度值相同，待拼接维度validShape等于tensor对应维度值，其余维度tensors validShape相同;

3.dim：-input.dim <= dim < input.dim（input对应tensors的任一tensor）；

4.设置viewshape时，dim对应维度不切块（即viewshape对应值\>=tensors任一tensor的对应值）；

5.输出Tensor的validShape需由用户在调用concat前确保正确，该接口不会自动推导。

6.Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入tensors维度为[m, c1, p]，[m, c2, p]，输出为[m, c1+c2, p]，TileShape设置为[m1, n1, p1]，则m1，p1分别用于切分m，p轴，n1用于切分c1和c2轴。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
a = pypto.ones([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 1s
b = pypto.zeros([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 0s
out = pypto.concat([a, b], dim = 0)
```

结果示例如下：

```python
输入数据a:   [[1.0 1.0],
              [1.0 1.0]]
输入数据b:   [[0.0 0.0],
              [0.0 0.0]]
输出数据out: [[1.0 1.0],
              [1.0 1.0],
              [0.0 0.0],
              [0.0 0.0]]

```
