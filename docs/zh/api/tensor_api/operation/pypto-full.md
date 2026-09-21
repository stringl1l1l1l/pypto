# pypto.full

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

创建一个大小为size、填充值为fill\_value的Tensor。其数据类型为dtype。

## 函数原型

```python
full(size: List[int], fill_value: Union[int, float, Element], dtype: DataType, *, valid_shape: Optional[Union[List[int], List[SymbolicScalar]]] = None ) -> Tensor
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| size         | 输入      | 源操作数，用于定义输出Tensor的Shape。<br>支持的数据类型为：List[int]。<br>支持的维度范围为：1维到4维。 |
| fill_value   | 输入      | 源操作数，用于填充输出Tensor的值。<br>支持的数据类型为：int，float，Element。<br>当为int或者float类型时会自动转换为Element类型，其中int对应DT_INT32，float对应DT_FP32。当需要使用其他数据类型时，可以通过Element构建。<br>Element支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>输入需要和dtype类型相同，不支持隐式转换。 |
| dtype        | 输入      | 源操作数，用于定义输出Tensor的类型。<br>支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>输入需要和fill_value类型相同，不支持隐式转换。 |
| valid_shape  | 输入      | 源操作数，用于定义输出Tensor的动态Shape，关键字参数，用于动态图，静态图可以省略。<br>支持的类型为List[SymbolicScalar]，List[int]。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和dtype相同，Shape为size大小，全部的值为fill\_value。

## 约束说明

1. valid\_shape用于动态图场景。

    在动态图场景中，若需生成 [5,5] 的Tensor并设置ViewShape为 [2,2]，框架会通过pypto.loop循环生成 [2,2] 分块，并按偏移量拼接。此时若未传入valid\_shape，代码将默认生成全 \[2,2\] 的Tensor（如pypto.full\(\[2,2\], 1, pypto.DT\_INT32\)）。

    然而，当总尺寸 \[5,5\] 无法被分块尺寸 \[2,2\] 整除时，尾块的有效形状（如 \[1,1\]）无法由框架自动推导。例如，最后一行/列可能仅包含1个元素，而非完整的 \[2,2\] 分块。此时必须通过valid\_shape明确指定尾块的实际有效形状，如下：

    pypto.full\(\[2, 2\], 1, pypto.DT\_INT32, valid\_shape=\[pypto.min\(2, 5 - 2 \* b\_idx\), pypto.min\(2, 5 - 2 \* s\_idx\)\]\)，其中b\_idx和s\_idx表示循环索引。

2. size的维度范围为1维到4维，即size的长度范围为\[1, 4\]。

3. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_BOOL。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_BOOL。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_BOOL。
   <!-- end id6 -->

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入size为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
# Valid shapes use keyword argument
x1 = 1.0 # must be 1.0; implicit conversion is not supported
y1 = pypto.full([2,2], x1, pypto.DT_FP32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

x2 = pypto.Element(pypto.DT_INT32,1)
y2 = pypto.full([2,2], x2, pypto.DT_INT32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

# In static graphs, validshape can be ignored
x3 = pypto.Element(pypto.DT_INT32,1)
y3 = pypto.full([2,2], x3, pypto.DT_INT32)
```

结果示例如下：

```python
y1输出数据: [[1.0,1.0], [1.0,1.0]]
y2输出数据: [[1,1], [1,1]]
y3输出数据: [[1,1], [1,1]]
```
