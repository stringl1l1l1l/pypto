# pypto.pad

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

填充大小根据`pad`参数从输入Tensor的最后一个维度开始，由后向前依次描述。`pad`参数的格式为 $(pad\_left, pad\_right, pad\_top, pad\_bottom, ...)$。当前实现仅支持对最后两个维度进行常量（Constant）模式的右侧（Right）和底部（Bottom）填充。

## 函数原型

```python
pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: Union[float, int] = 0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | 输入      | 需要进行填充的源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16、DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。                                                  |
| pad    | 输入      | 填充大小序列。<br>支持的类型为：tuple或list (包含int)。<br>序列长度 $m$ 必须为偶数，且满足 $\frac{m}{2} \leq$ `input`的维度数。<br>格式为：`(pad_left, pad_right, pad_top, pad_bottom, ...)`<br>所有填充大小序列的值必须为非负整数，负值不支持。                           |
| mode   | 输入      | 填充模式。<br>支持的类型为：str。<br>可选值为`'constant'`、`'reflect'`、`'replicate'`或`'circular'`。<br>默认值：`'constant'`。<br>**注意**：当前仅支持`'constant'`模式。                                             |
| value  | 输入      | 当填充模式为常量填充(`'constant'`)时的填充值。<br>支持的类型为：float或int。<br>对于浮点类型（DT_FP32、DT_FP16、DT_BF16），支持任意浮点数值，包括`-inf`、`inf`、`0.0`以及其他任意浮点数（如`1.0`、`-1.0`、`0.5`等）。<br>对于整型类型（DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32），value**仅支持整数值**，不支持传入`float('-inf')`或`float('inf')`，PyPTO会抛出`ValueError`提示用户传入实际整数值。<br>默认值：`0`。                                                                                                                                          |

## 返回值说明

返回输出Tensor，Tensor的数据类型和`input`相同，Shape为根据`pad`参数在对应维度上扩展后的大小。

## 约束说明

1. `pad`参数的长度必须为2或者4，`pad`参数中的填充大小序列的值必须为非负整数。负值填充不支持。如果传入负值，将抛出`ValueError`。
2. 当前**仅支持多维情况下在右侧（Right）和底部（Bottom）进行填充，或者1维情况下在右侧（Right）填充**。即`pad`序列中向左和向上的填充量必须为0（例如格式必须为`(0, pad_right, 0, pad_bottom)`或者`(0, pad_right)` ）。
3. mode当前**仅支持`'constant'`（常量填充）模式**，其他模式暂不支持。
4. **整型类型不支持浮点值**：对于整型dtype（DT_INT8、DT_INT16、DT_INT32、DT_UINT8、DT_UINT16、DT_UINT32），`value`参数不支持传入float类型的值，PyPTO会抛出`ValueError`。如需填充整型的最小/最大值，请显式传入对应dtype的实际值（如`int32`填`-2147483648`，`int16`填`-32768`）。
5. 如果`input`不是Tensor类型，或`pad`不是整数序列，将抛出`TypeError`。
6. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape维度应和**输出**一致。

示例1：输入`input` shape为`[m, n]`，如果对其在n轴右侧填充了`p`，则输出shape为`[m, n+p]`，TileShape设置为`[m1, n1]`，则`m1`，`n1`分别用于切分输出的`m`，`n+p`轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
# 示例：对一个shape为 [1, 1, 2, 2] 的Tensor进行填充
# 最后一个维度(右侧)填充1
# 倒数第二个维度(底部)填充1
t4d = pypto.tensor([0.0, 1.0, 2.0, 3.0], pypto.DT_FP32)
# 假设内部已将一维数据reshape为 [1, 1, 2, 2]

p1 = (0, 1, 0, 1)  # (pad_left=0, pad_right=1, pad_top=0, pad_bottom=1)
out = pypto.pad(t4d, p1, mode="constant", value=0.0)
```

结果示例如下：

```python
# 输入数据t4d (逻辑shape为 [1, 1, 2, 2]):
[[[[0.0, 1.0],
   [2.0, 3.0]]]]

# 输出数据out (逻辑shape扩展为 [1, 1, 3, 3]):
[[[[0.0, 1.0, 0.0],
    [2.0, 3.0, 0.0],
    [0.0, 0.0, 0.0]]]]
```
