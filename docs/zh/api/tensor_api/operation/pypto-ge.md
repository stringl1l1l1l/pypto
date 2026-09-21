# pypto.ge

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

逐元素大于等于比较运算。

## 函数原型

```python
ge(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。<br>支持的类型为：Tensor，float，Element。<br>当为float类型时会自动转换为Element类型，float对应DT_FP32。当需要使用其他数据类型时，可以通过Element构建。<br>不同型号支持的Tensor和Element的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>两个源操作数的数据类型必须保持一致。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值大于或者等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

## 约束说明

1. input和other类型须保持一致。
2. 支持多维度广播到相同形状。
3. Tensor和Element数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP16，DT_FP32，DT_INT16，DT_INT64，DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP16，DT_FP32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP16，DT_FP32
   <!-- end id6 -->
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.ge(a, b)
```

结果示例如下：

```python
输入数据a: [1.0 2.0 3.0]
输入数据b: [2.0 2.0 2.0]
输出数据out: [False, True, True]
```
