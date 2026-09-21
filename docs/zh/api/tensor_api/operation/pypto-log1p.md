# pypto.log1p

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

对1+input做以e为底的对数运算

## 函数原型

```python
log1p(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input    | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape为input大小。

## TileShape设置示例

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 约束说明

1. 在综合考虑输入、输出及临时存储空间占用的前提下，TileShape大小需满足额外约束条件。假设TileShape为\[a,b,c,d\]，记$d_{align}=CeilAlign(d, 8)$，则总的UB空间占用为：
   $$
   6*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log1p(x)
```

结果示例如下：

```python
输入数据x: [1e-99]
输出数据y: [1e-99]
```
