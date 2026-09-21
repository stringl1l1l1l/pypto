# pypto.gcd

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

求input与other对应元素的最大公约数。

## 函数原型

```python
gcd(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                                                   |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT8，DT_INT16，DT_INT32，DT_UINT8。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。           |
| other  | 输入      | 源操作数。<br>支持的类型为int以及Tensor类型。<br>Tensor支持的数据类型为：DT_INT8，DT_INT16，DT_INT32，DT_UINT8。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1. input和other数据类型相同。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape，TileShape尾轴须32Byte对齐。

TileShape维度应和输出一致。

示例1：非广播场景，输入input shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

示例2：广播场景，输入input shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)
y = pypto.tensor([2, 3], pypto.DT_INT32)
z = pypto.gcd(x, y)
# Using a scalar
c = pypto.gcd(x, 2)
```

结果示例如下：

```python
输入数据x: : [[9 9 9],
             [6 6 6]]
输入数据y:   [[1 2 3],
             [1 2 3]]
输出数据z:   [[1 1 3],
             [1 2 3]]
输出数据c:   [[1 1 1],
             [2 2 2]]
```
