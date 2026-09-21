# pypto.fmod

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

将input的每个元素和other中对应位置的元素进行取模运算，计算公式如下：

$$
res_i = input_i \;\%\; other_i
$$

## 函数原型

```python
fmod(input: Tensor, other: Union[Tensor, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维，并支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。<br>支持的类型为float以及Tensor类型。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维，并支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | 输入 | 精度模式枚举类型，用以控制取模计算的精度模式，具体定义为：[PrecisionType](../datatype/PrecisionType.md)。<br>默认为HIGH_PRECISION（高精度模式）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1. input和other数据类型应相同。
2. other为数字的时候，不支持隐式转换。
3. other不支持nan、inf等特殊值。
4. precision_type使用说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：支持高精度模式
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：不支持高精度模式，默认使用指令模式 `INTRINSIC`。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：不支持高精度模式，默认使用指令模式 `INTRINSIC`。
   <!-- end id6 -->
5. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b)
```

结果示例如下：

```python
输入数据a:    [[7.0 8.0 9.0]]
输入数据b:    [[3.0 3.0 3.0]]
输出数据out:  [[1.0 2.0 0.0]]
```

### 高精度模式示例

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b, pypto.PrecisionType.INTRINSIC)
```
