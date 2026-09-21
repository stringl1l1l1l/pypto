# pypto.expm1

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

将输入Tensor的每个元素进行自然指数计算后减1
$$
y_i = e^{x_i} - 1
$$

## 函数原型

```python
expm1(input: Tensor) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                                                                                                             |
|----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input    | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape与输入Tensor一致。当输入数据类型为DT_FP32，DT_FP16，DT_BF16时，其数据类型与输入Tensor一致；当输入数据类型为DT_INT32，DT_INT16时，其数据类型为DT_FP32。其元素为输入Tensor对应元素进行自然指数计算后减1。

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
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.expm1(x)
```

结果示例如下：

```python
输入数据x: [[1., 2.], [3., 4.]]
输出数据y: [[1.7183, 6.3891], [19.0855, 53.5981]]
```
