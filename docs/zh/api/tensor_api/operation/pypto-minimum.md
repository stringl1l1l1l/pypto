# pypto.minimum

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

计算输入与另一输入的最小值。支持二维、三维或四维的Tensor。

## 注意事项

- **不支持SymbolicScalar参数**：如果需要对SymbolicScalar进行比较，请使用[SymbolicScalar.min()](../symbolic/pypto-SymbolicScalar-min.md)方法
- 两个参数中至少一个为Tensor类型

## 函数原型

```python
minimum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为int，float，Element以及Tensor类型。<br>当为int或者float类型时会自动转换为Element的类型DT_INT32/DT_FP32。当需要使用其他数据类型时，可以通过Element构建。<br>不同型号支持的Tensor和Element数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。<br>支持的类型为int，float，Element以及Tensor类型。<br>当为int或者float类型时会自动转换为Element的类型DT_INT32/DT_FP32。当需要使用其他数据类型时，可以通过Element构建。<br>不同型号支持的Tensor和Element数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。<br>类型和数据类型必须与源操作数一保持一致。 |

源操作数一与源操作数二之间至少一者为Tensor。

## 返回值说明

当两个源操作数均为Tensor时，两个Tensor必须满足广播关系。该接口返回一个与源操作数一和源操作数二广播后Shape相同的Tensor，数据类型与源操作数相同，其元素为源操作数一和源操作数二的逐元素最小值。且源操作数为Tensor时，源操作数一和源操作数二均支持多轴广播。

当两个源操作数之中存在一个Tensor时，返回与输入Tensor相同Shape的Tensor，其元素为源操作数一和源操作数二的逐元素最小值。

## 约束说明

1. 两个输入均为Tensor类型时，支持的数据类型如下：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT32，DT_UINT32，DT_FP32，DT_INT16，DT_UINT16，DT_FP16，DT_BF16，DT_UINT8，DT_INT8，DT_INT64，DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16
   <!-- end id6 -->
2. 一个输入为Tensor，另一个输入为Element类型时，支持的数据类型如下：
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16，DT_INT64，DT_UINT64
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16
   <!-- end id9 -->
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
a = pypto.tensor([3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
out = pypto.minimum(a, b)
```

结果示例如下：

```python
输入数据a: [0, 2, 4]
输入数据b: [3, 1, 3]
输出数据out: [0, 1, 3]
```
