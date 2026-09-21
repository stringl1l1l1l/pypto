# pypto.prod

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

对一个多维向量按照指定的维度进行数据累乘。

## 函数原型

```python
prod(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_INT32，DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维，Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定进行归约的维度。<br>支持任意单轴。 |
| keepdim | 输入      | 归约后是否保留被归约的维度。<br>默认值为False。 |

## 返回值说明

返回输出Tensor，输出Tensor的Shape与keepdim参数相关。

若keepdim参数为True，则在执行归约操作后保留被归约的维度。输出Tensor在除dim指定的维度外，其他维度的Shape与输入Tensor的Shape一致，而在dim指定的维度上的大小为1。

若keepdim参数为False（默认），则被归约的维度会从输出Tensor中移除，而tileshape中对应的维度不变，所以建议在调其他operation前重设tileshape。

## 约束说明

1. TileShape大小不超过64KB；

2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## TileShape设置示例

TileShape维度应和输入input一致。

如输入input shape为[m, n]，输出为[m, 1]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.prod(x, -1, True)
```

结果示例如下：

```txt
输入数据x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
输出数据y: [[6.0],
             [6.0]]
```
