# pypto.where

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

condition为一个布尔类型的掩码张量（mask tensor）。对于张量中任意位置的元素，该操作基于布尔掩码张量condition进行逐元素选择。其计算行为可形式化表示为如下表达式。

$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

condition须为Tensor，input和other可以为Tensor、float以及Element，广播规则如下（只支持单轴广播）：

1. input，other，condition均为Tensor，result的Shape由三者广播得到。

    例：input：\[1,20,20\]，other：\[20,1,20\]，condition：\[20,20,1\]，result：\[20,20,20\]

2. 只有input，condition为Tensor时，result的Shape由两者广播得到。

    例：input：\[1,20,20\]，condition：\[20,20,1\]，result：\[20,20,20\]

3. 只有other，condition为Tensor时，result的Shape由两者广播得到。

    例：other：\[20,1,20\]，condition：\[20,20,1\]，result：\[20,20,20\]

4. 只有condition为Tensor时，result的Shape与condition一致。

## 函数原型

```python
where(
    condition: Tensor,
    input: Union[Tensor, float, Element],
    other: Union[Tensor, float, Element]
) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| condition   | 输入      | 支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_BOOL。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。<br>作为条件选择input或者other的元素。 |
| input       | 输入      | 支持的类型为 float/Element/Tensor类型。<br>当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。<br>不同型号支持的Tensor和Element数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other       | 输入      | 支持的类型为 float/Element/Tensor类型。<br>当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。<br>不同型号支持的Tensor和Element数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

result ：Tensor，Shape由输入的广播得到，详细广播场景可看上文。数据类型和input、other保持一致。

## 约束说明

1. 当input或other为float标量时，float会自动转换为DT_FP32的Element。若另一个参数为Tensor且其数据类型不是DT_FP32（如DT_FP16、DT_BF16等），将触发类型不一致拦截报错，不支持此种用法。请使用Element构建与Tensor一致的数据类型，例如：`pypto.Element(pypto.DT_FP16, 1.0)`。
2. condition、input（如果是tensor）、other（如果是tensor）的维度必须相同。例如condition：[64]，input：[2, 64]，other：[2, 64]这种情况非法，应当设置为condition：[1, 64]，input：[2, 64]，other：[2, 64]。
3. input和other的Tensor、Element数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT32，DT_FP32，DT_INT16，DT_FP16，DT_BF16，DT_UINT8，DT_INT8，DT_INT64，DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16
   <!-- end id6 -->
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：非广播场景，输入condition为[m, n]，input为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

示例2：广播场景，输入condition为[m, 1]，input为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
cond1 = pypto.tensor([4], pypto.DT_BOOL)
a1 = pypto.tensor([4], pypto.DT_FP32)
b1 = pypto.tensor([4], pypto.DT_FP32)
out1 = pypto.where(cond1, a1, b1)

# Using scalar inputs
out2 = pypto.where(cond1, 1, 0)

# Broadcasting example
cond2 = pypto.tensor([2, 2], pypto.DT_BOOL)
a2 = pypto.tensor([2], pypto.DT_FP32)
b2 = 0.0
out3 = pypto.where(cond2, a2, b2)
```

结果示例如下：

```python
输入数据cond1: [True, False, True, False]
输入数据a1:    [1.0  2.0  3.0  4.0]
输入数据b1:    [10.0 20.0 30.0 40.0]
输出数据out1:  [1.0  20.0 3.0  40.0]

输出数据out2:  [1.0 0.0 1.0 0.0]

输入数据cond2 = [[True, False], [False, True]]
输入数据a2:      [1.0 2.0]
输出数据out3:   [[1.0 0.0],
                 [0.0 2.0]]
```
