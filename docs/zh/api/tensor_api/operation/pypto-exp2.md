# pypto.exp2

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

计算输入Tensor中每个元素的2的指数，逐元素运算，返回与输入形状相同的Tensor。

## 函数原型

```python
exp2(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16，DT_INT8，DT_UINT8。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，当输入是DT_FP32，DT_FP16，DT_BF16，Tensor的数据类型和input相同，Shape与input相同，当输入是DT_INT32，DT_INT16，DT_INT8，DT_UINT8，Tensor的数据类型是DT_FP32，Shape与input相同。

## 约束说明

1. input中的值域范围需要在\[-2^24, 2^24\]范围内，以确保在计算过程中能精确转换为float32。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
3. 临时空间使用说明：若TileShape为\[a,b,c,d\]则需要的临时空间大小为
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：a\*b\*c\*d\*sizeof\(DT_FP32\)
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id6 -->

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
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp2(x)
```

结果示例如下：

```python
输入数据x: [0.0    1.0    2.0]
输出数据y: [1.0    2.0    4.0]
```
