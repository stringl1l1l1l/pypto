# pypto.sin

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

计算输入Tensor中每个元素的正弦值（三角函数sin\( \)），逐元素运算。

## 函数原型

```python
sin(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：DT_FP32，DT_FP16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入Tensor对应元素的正弦值。

## 约束说明

1. 输入Tensor和输出Tensor类型应该相同。
2. 由于存在临时内存使用，TileShape大小有额外约束，假设TileShape为\[a,b,c,d\]，那么5\*a\*b\*c\*d\*sizeof\(DT_FP32\) < UB。
3. input支持的输入范围为[-65504.0,65504.0]。
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
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sin(x)
```

结果示例如下：

```python
输入数据x: [-0.5461, 0.1347, -2.7266, -0.2746]
输出数据y: [-0.5194, 0.1343, -0.4032, -0.2711]
```
