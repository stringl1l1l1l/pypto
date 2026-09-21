# pypto.ones

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

创建一个大小为`size`、填充值全为`1`的Tensor。其数据类型由`dtype`指定，默认数据类型为`DT_FP32`。

## 注意事项

- **必须先设置TileShape**：调用此接口前，必须先通过[set_vec_tile_shapes](../config/pypto-set_vec_tile_shapes.md)设置TileShape
- **dtype参数必须显式传入**：当需要指定数据类型时，必须使用关键字参数`dtype=`显式传入，不能作为位置参数传入。例如，应使用`pypto.ones(2, 3, dtype=pypto.DT_INT32)`而非`pypto.ones(2, 3, pypto.DT_INT32)`。如果作为位置参数传入，dtype值会被误解析为size的一个维度，导致错误

## 函数原型

```python
ones(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| *size        | 输入      | 源操作数，用于定义输出Tensor的Shape。<br>支持可变长参数（多个int）或单一的序列（如List[int] 或Tuple[int]）。 |
| dtype        | 输入      | 源操作数，可选参数，用于定义输出Tensor的数据类型。<br>支持的数据类型为：`DT_FP32`，`DT_INT32`，`DT_INT16`，`DT_FP16`，`DT_BF16`。<br>默认值为`pypto.DT_FP32`。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型由`dtype`决定，Shape为`size`大小，全部的值均为`1`。

## 约束说明

1. `TileShape`的维度需要与输出result维度相同，用于切分result。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。TileShape维度应和输出一致。
如输入size为`[m, n]`，输出为`[m, n]`，TileShape设置为`[m1, n1]`，则`m1`，`n1`分别用于切分`m`，`n`轴。

```python
pypto.set_vec_tile_shapes(2, 3)
```

### 接口调用示例

```python
# 示例1：使用可变参数传入size，使用默认dtype (DT_FP32)
x1 = pypto.ones(2, 3)

# 示例2：使用列表传入size，显式指定dtype (DT_INT32)
x2 = pypto.ones([2, 3], dtype=pypto.DT_INT32)
```

结果示例如下：

```python
x1输出数据: [[1., 1., 1.],
             [1., 1., 1.]]
x2输出数据: [[1, 1, 1],
             [1, 1, 1]]
```
