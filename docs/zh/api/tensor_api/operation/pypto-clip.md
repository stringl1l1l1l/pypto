# pypto.clip

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

对输入Tensor进行数据裁剪，裁剪到指定的最小值到最大值范围内，小于最小值的位置替换为最小值，大于最大值的位置替换为最大值，其余值维持不变。该接口非原地操作，不改变输入Tensor，而是返回一个新的Tensor作为输出。

## 函数原型

```python
clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None
)-> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为Tensor类型。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor，数据维度大小仅支持1-4维，元素个数不超过INT32_MAX。 |
| min    | 输入      | 源操作数。<br>支持的类型为int\float\Element以及Tensor类型，数据类型必须与输入input的数据类型一致。<br>当为int或者float类型时会自动转换为Element类型，且数据类型与输入input的数据类型一致。当需要使用其他数据类型时，可以通过Element构建。<br>Tensor和Element支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor，数据维度大小仅支持1-4维，元素个数不超过INT32_MAX。<br>可缺省，默认值为-INF。<br>NaN，INF，-INF仅在浮点数运算时有定义，即只在数据类型为DT_FP16/DT_FP32时生效；当数据类型为DT_INT16和DT_INT32时，会跳过默认值的比较逻辑。 |
| max    | 输入      | 源操作数。<br>支持的类型为int\float\Element以及Tensor类型，数据类型必须与输入input的数据类型一致。<br>当为int或者float类型时会自动转换为Element类型，且数据类型与输入input的数据类型一致。当需要使用其他数据类型时，可以通过Element构建。<br>Tensor和Element支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor，数据维度大小仅支持1-4维，元素个数不超过INT32_MAX。<br>可缺省，默认值为INF。<br>NaN，INF，-INF仅在浮点数运算时有定义，即只在数据类型为DT_FP16/DT_FP32时生效；当数据类型为DT_INT16和DT_INT32时，会跳过默认值的比较逻辑。 |

## 返回值说明

当输入为标量时，输出为：

$$
Y_{i} = \text{MIN}\left( \text{MAX}\left(X_{i}, \text{min\_value}\right), \text{max\_value} \right)
$$

当输入为Tensor时，输出为：

$$
Y_{i} = MIN\left( MAX(X_{i}, min\_value_{i}), max\_value_{i} \right)
$$

输出Tensor的数据类型和输入input相同。

当min / max其中一者为NAN时，输出结果为NAN。

当min \> max时，输出结果对应位置均为max的值。

## 约束说明

1. min / max的类型必须一致，同时为Element或同时为Tensor。
2. min / max为Tensor类型时，其Shape大小必须满足可以广播到输入的Shape。
3. min和max支持同时缺省，返回原值。
4. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16，DT_INT64。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。
   <!-- end id6 -->
5. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入input shape为[m, n]，max和min为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

广播场景，输入input shape为[m, n]，max和min为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = pypto.tensor([2,3], pypto.DT_INT32)
max = pypto.tensor([2,3], pypto.DT_INT32)
out = pypto.clip(x,min,max)
```

结果示例如下：

```python
输入数据self: [[-2 1 2], [3 4 5]]
输入数据min: [[-1 0 2], [0 3 5]]
输入数据max: [[1 2  1], [4 4 4]]
输出数据out: [[-1 1 1], [3 4 4]]
```

示例2：

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = 1
max = 3
out = pypto.clip(x,min,max)
```

结果示例如下：

```python
输入数据x: [[0 2 4], [3 4 6]]
输出数据out: [[1 2 3], [3 3 3]]
```
