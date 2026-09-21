# pypto.ceil_div

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

将self的每个元素除以other中对应位置的元素并向上取整，计算公式如下：

$$
res_i = ceil(self_i \div other_i)
$$

## 函数原型

```python
ceil_div(self: Tensor, other: Union[Tensor, int]) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| self  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT32。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入     | 源操作数。<br>支持的类型为：Tensor、int。<br>Tensor支持的数据类型为：DT_INT32。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和self、other相同，Shape为self和other广播后大小。

## 约束说明

1. self和other类型应该相同。
2. 支持多维度广播。
3. self和other中的值域范围需要在\[-2^24, 2^24\]范围内，以确保在计算过程中能精确转换为float32，**other不能为0**，整数除法为0时结果由芯片决定，可能为INT32_MAX或INT32_MIN。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入self shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

广播场景，输入self shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
a = pypto.tensor([1, 3], pypto.DT_INT32)
b = pypto.tensor([1, 3], pypto.DT_INT32)
out = pypto.ceil_div(a, b)
```

结果示例如下：

```python
输入数据a:    [[2 4 6]]
输入数据b:    [[4 2 5]]
输出数据out:  [[1 2 2]]
```
