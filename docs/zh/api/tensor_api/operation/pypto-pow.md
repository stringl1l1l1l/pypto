# pypto.pow

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

计算输入Tensor中每个元素的other次幂，逐元素运算，返回与输入形状相同的Tensor。

## 函数原型

```python
pow(input: Tensor, other: Union[Tensor, int, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16、DT_BF16、DT_FP32、DT_INT32、DT_INT8、DT_UINT8、DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 指数。<br>支持的类型为Tensor、int或float。<br>Tensor支持的数据类型为：DT_FP16、DT_BF16、DT_FP32、DT_INT32、DT_INT8、DT_UINT8、DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | 输入      | 精度模式枚举类型，用以控制指数计算的精度模式，具体定义为：[PrecisionType](../datatype/PrecisionType.md) 。<br>默认为HIGH_PRECISION（高精度模式）。 |

## 返回值说明

返回一个与输入形状相同的Tensor，其元素为输入Tensor对应元素的other次幂。

当other为int时，返回的Tensor的数据类型与输入相同。

当other为float时，若输入Tensor类型为DT_INT32则返回DT_FP32，否则返回的Tensor的数据类型与输入相同。

当other为Tensor时，返回的Tensor的数据类型见数据类型提升说明章节。

## 约束说明

1. precision_type使用说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：支持
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：不支持，默认使用指令模式 `INTRINSIC`
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：不支持，默认使用指令模式 `INTRINSIC`
   <!-- end id6 -->
2. 两个输入均为Tensor且输入类型为int8/uint8/int16时，两个输入参数数据类型需相同。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
4. 临时空间使用说明：若TileShape为\[a,b,c,d\]，当输出数据类型为DT_INT32、DT_INT8、DT_UINT8、DT_INT16时需要临时空间大小为a\*b\*c\*d\*sizeof\(DT_FP32\)，其余情况需要的临时空间大小为
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：a\*b\*c\*d\*sizeof\(DT_FP32\)
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3系列产品：a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2系列产品：a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id9 -->

## 数据类型提升说明

我们约定float32>float16>bfloat16>int32。

1. 当两个输入参数类型均为int8/uint8/int16时，输出类型与输入一致。
2. 当两个输入参数类型一个为float16而另一个bfloat16时输出的数据类型为float32。
3. 其他情况下输出类型为输入参数类型的更大值，如输入float32和float16则输出为float32，参考下述表格。

| 参数类型    | float32    | float16    | bfloat16   | int32      |
|-------------|------------|------------|------------|------------|
| **float32**     | float32    | float32    | float32    | float32    |
| **float16**     | float32    | float16    | **float32**    | float16    |
| **bfloat16**    | float32    | **float32**    | bfloat16   | bfloat16   |
| **int32**       | float32    | float16    | bfloat16   | int32      |

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
a = 2
b = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.pow(x, a)
z = pypto.pow(x, b)
```

结果示例如下：

```python
输入数据x: [[1.0  2.0], [-3.0  4.0]]
输入数据b: [[2.0  2.0], [1.0   1.0]]
输出数据y: [[1.0  4.0], [9.0  16.0]]
输出数据z: [[1.0  4.0], [-3.0  4.0]]
```

### 高精度模式示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.tensor([2, 2], pypto.DT_FP16)
out = pypto.pow(x, y, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.pow(x, y, pypto.PrecisionType.INTRINSIC)
```
