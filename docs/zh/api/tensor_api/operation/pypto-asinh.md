# pypto.asinh

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

逐元素计算输入Tensor的反双曲正弦值。

$$
y_i = \operatorname{asinh}(x_i) = \ln(x_i + \sqrt{x_i^2 + 1})
$$

## 函数原型

```python
asinh(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Shape与`input`相同，数据类型与`input`相同，元素值为输入Tensor对应元素的反双曲正弦值。

## 约束说明

1. 在综合考虑输入、输出及临时存储空间占用的前提下，TileShape大小需满足额外约束条件。假设TileShape为\[a,b,c,d\]，记 $d_{align}=CeilAlign(d, 8)$，$k=d_{align}/8$，$p=\lceil8/k\rceil$，$c_{pad}=c+p-1$，则总的UB空间占用为：

   $$
   a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)+5*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

TileShape维度应和输出一致。

如输入`input` shape为`[m, n]`，输出为`[m, n]`，TileShape设置为`[m1, n1]`，则`m1`、`n1`分别用于切分`m`、`n`轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.asinh(x)
```

结果示例如下：

```python
输入数据x: [0.0000, 1.0000, 2.0000, -1.0000]
输出数据y: [0.0000, 0.8814, 1.4436, -0.8814]
```
