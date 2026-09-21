# pypto.expand\_exp\_dif

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

计算自然对数的底数e的(input - other)次幂，返回与输入input广播后形状相同的Tensor。计算公式如下：

$$
res_i = e^{(input_i - other_i)}
$$

## 函数原型

```python
expand_exp_dif(input: Tensor, other: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_FP32，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_FP32，DT_BF16。<br>不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型与input、other相同，Shape为input和other广播后大小。

## 约束说明

1. input和other都为Tensor时，数据类型应该相同。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入input shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

广播场景，输入input shape为[m, n]，other为[1, n]或[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

#### 示例1：尾轴广播

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 1], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

结果示例如下：

```python
输入数据x:     [[1, 2, 3], [4, 5, 6]]
输入数据y:     [[1], [2]]
输出数据out:   [[ 1.       ,  2.718282 ,  7.3890557],
               [ 7.3890557, 20.085537 , 54.59815  ]]
```

#### 示例2：多轴广播

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([1, 1], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

结果示例如下：

```python
输入数据x:     [[1, 2, 3], [4, 5, 6]]
输入数据y:     [[1]]
输出数据out:   [[ 1.       ,  2.718282 ,  7.3890557],
               [20.085537 , 54.59815  , 148.41316 ]]
```
