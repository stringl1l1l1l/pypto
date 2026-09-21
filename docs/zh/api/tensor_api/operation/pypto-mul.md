# pypto.mul

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

按元素求积，公式表达如下：

$$
res_{i} = input_{i} * other_{i}
$$

## 函数原型

```python
mul(input: Tensor, other: Union[Tensor, float, int]) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。<br>支持的类型为float、int以及Tensor类型。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1. input和other都为Tensor时，数据类型应该相同。
2. other为scalar时，若input为浮点类型，则scalar支持整型（自动转为浮点）；若input为整型，则scalar不支持浮点类型（会报错）。
3. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT32，DT_FP32，DT_INT16，DT_FP16，DT_BF16，DT_INT64，DT_UINT64。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id6 -->
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入input shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

广播场景，输入input shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 3], pypto.DT_FP32)
z = pypto.mul(x, y)
```

结果示例如下：

```python
输入数据x:  [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
输入数据y:  [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
输出数据z:  [[1.0 4.0 9.0],
             [1.0 4.0 9.0]]
```
