# pypto.sinh

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

计算输入Tensor中每个元素的双曲正弦值，逐元素运算。

$$
y_i = \sinh(x_i) = \frac{e^{x_i} - e^{-x_i}}{2}
$$

## 函数原型

```python
sinh(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。|

## 返回值说明

返回Tensor类型。其Shape与输入Tensor一致，数据类型与输入Tensor一致，其元素为输入Tensor对应元素的双曲正弦值。

## 约束说明

1. 在综合考虑输入、输出及临时存储空间占用的前提下，TileShape大小需满足额外约束条件。假设TileShape为\[a,b,c,d\]，记$d_{align}=CeilAlign(d, 8)$，$k=d_{align}/8$，$p=\lceil8/k\rceil$，$c_{pad}=c+p-1$，则总的UB空间占用为：

   $$
   a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)+5*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sinh(x)
```

结果示例如下：

```python
输入数据x: [0.0000, 1.0000, 2.0000, -1.0000]
输出数据y: [0.0000, 1.1752, 3.6269, -1.1752]
```
