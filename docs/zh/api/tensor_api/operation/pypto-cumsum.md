# pypto.cumsum

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

计算输入Tensor input沿指定维度的累积和。

## 函数原型

```python
cumsum(input: Tensor, dim: int) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                              |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input  | 输入      | 源操作数。支持的类型为：Tensor。Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_INT16，DT_INT32，DT_FP32。不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim    | 输入      | 指定进行累加的维度。int类型。                                                                                                                                                                       |

## 返回值说明

输出Tensor Shape与输入input一致。
input为DT_FP16，DT_BF16，DT_FP32等类型时，输出数据类型和输入input一致，input为DT_INT16，DT_INT32时，输出数据类型为DT_INT64。

## 约束说明

1. dim：指定计算累积和的维度，必须在输入Tensor input的有效维度范围内，其值需满足-input.dim <= dim < input.dim；
2. input的dim轴ViewShape不可切分，其余维度不做限制；
3. TileShape的维度与input相同，所有输入和输出的TileShape大小总和不能超过UB内存的大小。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
input = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
dim = 0
y = pypto.cumsum(input, dim)
```

结果示例如下：

```python
输入数据x:   [[0 1 2],
               [3 4 5]]
输出数据y:   [[0 1 2],
               [3 5 7]]                             # shape (2, 3)
```
