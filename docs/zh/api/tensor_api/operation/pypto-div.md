# pypto.div

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

将input的每个元素除以other中对应位置的元素，计算公式如下：

$$
res_i = input_i \div other_i
$$

## 函数原型

```python
div(input: Tensor, other: Union[Tensor, float, int], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；支持的维度：1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。<br>支持的类型为：Tensor、float、int。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；支持的维度：1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | 输入 | 精度模式枚举类型，用以控制除法计算的精度模式，具体定义为：[PrecisionType](../datatype/PrecisionType.md)。<br>默认为HIGH_PRECISION（高精度模式）。 |

## 返回值说明

返回输出Tensor，Shape为input和other广播后大小。
当input为浮点类型时，输出Tensor数据类型与input相同；当input为整型时，输出数据类型为DT_FP32。

## 约束说明

1. input和other都为Tensor时，数据类型应该相同；input为整型类型时，other暂不支持浮点标量。
2. Tensor数据类型说明：
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：DT_FP16，DT_FP32，DT_BF16，DT_INT16，DT_INT32，DT_INT64。
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3系列产品：DT_FP16，DT_FP32，DT_BF16，DT_INT16，DT_INT32。
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2系列产品：DT_FP16，DT_FP32，DT_BF16，DT_INT16，DT_INT32。
   <!-- end id9 -->
3. **精度模式说明**：
    - **HIGH_PRECISION（高精度模式）**：默认模式，在底层实现中会使用更高精度的计算方式，在不同型号上的支持情况：
      <!-- npu="950" id4 -->
      - Ascend 950PR&950DT系列产品：支持
      <!-- end id4 -->
      <!-- npu="A3" id5 -->
      - Atlas A3系列产品：不支持
      <!-- end id5 -->
      <!-- npu="910b" id6 -->
      - Atlas A2系列产品：不支持
      <!-- end id6 -->
    - **INTRINSIC（指令模式）**：直接使用芯片指令进行计算。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
5. 整型输入约束：
当输入为DT_INT16或DT_INT32或DT_INT64时，内部会将输入转换为DT_FP32进行计算（float32尾数为24位）。在 $[-2^{24},\ 2^{24}]$ 范围内的整数可精确转换，超出范围的整数在转换时可能丢失低位精度。

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

#### 基本用法（默认使用高精度模式）

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b)  # 默认使用HIGH_PRECISION模式
```

结果示例如下：

```python
输入数据a:    [[2.0 4.0 6.0]]
输入数据b:    [[2.0 2.0 2.0]]
输出数据out:  [[1.0 2.0 3.0]]
```

#### 显式指定高精度模式

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

#### 使用指令模式

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)
```

#### 使用运算符（自动使用高精度模式）

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = a / b  # 自动使用HIGH_PRECISION模式
out = a.div(b)  # 自动使用HIGH_PRECISION模式
```
