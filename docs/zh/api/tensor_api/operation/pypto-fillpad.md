# pypto.fillpad

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

对输入Tensor进行填充（Padding）。

和pad不同，此接口不会改变张量的形状，它将填充区域（即超过validshape的区域）用指定的值进行填充。当前实现支持输入1-2维Tensor，进行常量（Constant）模式的右侧（Right）和底部（Bottom）填充。

## 函数原型

```python
fillpad(input: Tensor, mode: str = "constant", value: Union[float, int] = 0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | 输入      | 需要进行填充的源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16、DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32。<br>不支持空Tensor；Shape支持1-2维；Shape Size不大于2147483647（即INT32_MAX）。                                                     |
| mode   | 输入      | 填充模式。<br>支持的类型为：str。<br>可选值为`'constant'`、`'reflect'`、`'replicate'`或`'circular'`。<br>默认值：`'constant'`。<br>**注意**：当前仅支持`'constant'`模式。                                             |
| value  | 输入      | 当填充模式为常量填充(`'constant'`)时的填充值。<br>支持的类型为：float或int。<br>对于浮点类型（DT_FP32、DT_FP16、DT_BF16），支持任意浮点数值，包括`-inf`、`inf`、`0.0`以及其他任意浮点数（如`1.0`、`-1.0`、`0.5`等）。<br>对于整型类型（DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32），支持任意整数值。<br>默认值：`0`。|

## 返回值说明

返回输出Tensor，Tensor的数据类型和`input`相同，Shape也和`input`相同。

## 约束说明

1. mode当前**仅支持`'constant'`（常量填充）模式**，其他模式暂不支持。
2. value支持任意浮点数值或整数值，填充值的数据类型会自动转换为与输入Tensor一致。
3. 如果`input`不是Tensor类型，将抛出`TypeError`。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape维度应和**输出**一致。

示例1：输入`input` shape为`[m, n]`，则输出shape为`[m, n]`，TileShape设置为`[m1, n1]`，则`m1`，`n1`分别用于切分输出的`m`，`n`轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
a = pypto.tensor([4, 4], pypto.DT_FP32)
out = pypto.fillpad(a, "constant", "-inf")
```

结果示例如下：

```python
# 输入数据t4d (逻辑shape为 [4, 4]):
[[1.0, 2.0, 0.0, 0.0],
[3.0, 4.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0]]

# 输出数据out (逻辑shape为 [4, 4]):
[[1.0, 2.0, -inf, -inf],
[3.0, 4.0, -inf, -inf],
[-inf, -inf, -inf, -inf],
[-inf, -inf, -inf, -inf]]
```
