# pypto.normal

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

生成指定shape的正态分布（高斯分布）随机数，其元素服从均值为0，方差为1。
$$
x_i \sim N(0, 1)
$$

## 函数原型

```python
normal(shape: List[int], key: List[int], counter: List[int], alg: List[int], dtype: DataType) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|--------------------------------------------------------------------|
| shape   | 输入      | 输出Tensor的形状。<br>数组长度支持1-4。                                        |
| key     | 输入      | 随机数生成器的seed。<br>数组长度仅支持1。                                         |
| counter | 输入      | 随机数生成器的计数器。<br>数组长度仅支持2。                                          |
| alg     | 输入      | 随机数生成算法，当前仅支持值1（Philox算法），3（auto_select，选择Philox算法）。<br>数组长度仅支持1。 |
| dtype   | 输入      | 输出Tensor的数据类型。<br>支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。            |

## 约束说明

- 不支持shape切分多个view shape，view shape必须和输入的shape一致。
- 不支持shape切分多个tile shape，tile shape必须和输入的shape一致。
- tile shape尾轴必须是4的倍数。
- `counter[0]`在内部被硬编码为0。虽然接口接受长度为2的counter列表，但`counter[0]`的值会被忽略，实际使用的Philox计数器为`[0, counter[1]]`。

## 返回值说明

返回一个指定shape、数据类型为dtype的Tensor，其元素服从均值为0，方差为1的正态分布。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入一致，且必须和输入的shape一致。

如输入shape为[m, n]，输出为[m, n]，TileShape设置为[m, n]。

```python
pypto.set_vec_tile_shapes(4, 4)
```

### 接口调用示例

```python
shape = [4, 4]
key = [1234]
counter = [0, 1]
alg = [1]
dtype = pypto.DT_FP32

y = pypto.normal(shape, key, counter, alg, dtype)
```

结果示例如下：

```python
输出数据y: [[-0.32364845  1.8577391   0.39556974  0.2311697 ]
            [ 0.24243996 -1.9485782  -0.12983137  2.7137496 ]
            [ 1.6558666   2.0938187  -0.90338254  0.8765667 ]
            [ 0.86518306  0.01034508  0.2893259   0.01748212]]
```
