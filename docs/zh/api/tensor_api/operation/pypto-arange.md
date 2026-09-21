# pypto.arange

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

创建长度为$\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil$的一维Tensor，包含区间 \[start, end\)内、以step为步长的等差数列。

## 函数原型

```python
arange(start: Union[int, float] = 0, end: Union[int, float], step: Union[int, float] = 1) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| start  | 输入      | 源操作数。<br>支持的数据类型为int、float。<br>默认值为0。 |
| end    | 输入      | 源操作数。<br>支持的数据类型为int、float。<br>该参数不能省略。 |
| step   | 输入      | 源操作数。<br>支持的数据类型为int、float，默认值为1。<br>不能为0，作为浮点数时，abs\(step\)\>1e-8。<br>\(end-start\)/step需大于0。 |

## 返回值说明

返回一维输出Tensor。

- 若输入值存在float数据类型，则输出Tensor数据类型为DT_FP32；
<!-- npu="950" id7 -->
- 对于Ascend 950PR&950DT系列产品，若输入值为int数据类型且超出DT_INT32范围，则输出Tensor数据类型为DT_INT64；
<!-- end id7 -->
- 对于其他情况，输出Tensor数据类型为DT_INT32。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape和输出output维度一致，均为一维。

如输入start为m，end为n，step为p，输出shape为[q]，TileShape设置为[q1]，则q1分别用于切分q轴。

```python
pypto.set_vec_tile_shapes(16)
```

### 接口调用示例

```python
y1 = pypto.arange(1.0, 4.0, 0.5)
y2 = pypto.arange(1.0, 4.0)
y3 = pypto.arange(4)
```

结果示例如下：

```python
输出数据y1: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
输出数据y2: [1.0, 2.0, 3.0]
输出数据y3: [0, 1, 2, 3]
```
