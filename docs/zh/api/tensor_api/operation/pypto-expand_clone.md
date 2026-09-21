# pypto.expand\_clone

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

将输入Tensor在等于1的轴上广播以匹配Shape，返回真实占内存的新Tensor。

## 函数原型

```python
expand_clone(
    input: Tensor,
    shape: List[int],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_BF16，DT_FP32，DT_FP16，DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_BOOL。<br>不支持空Tensor；Shape仅支持1-4维，被广播的轴的Shape大小要为1；Shape Size不大于2147483647（即INT32_MAX）。 |
| shape       | 输入      | 目标Shape。<br>支持的数据类型为：List[int]。<br>Shape Size不大于INT32_MAX；Shape的维度需要与输入的一致，除被广播的轴外其他轴大小须与input的Shape对应相等。 |
| valid_shape | 输入      | 关键字参数。<br>源操作数，用于定义输出Tensor的动态Shape，关键字参数，用于动态图，静态图可以省略。<br>支持的类型为List[SymbolicScalar]，List[int]。 |

## 返回值说明

返回输出Tensor，其数据类型和input相同，形状为shape。

## 约束说明

1. 支持多维广播，输入Tensor被广播的轴的Shape大小要为1。
2. input的viewshape与input维度相同，viewshape\[dim\]=1，input\[dim\]=1，其中dim为被拓展轴，其余维度不做限制。举例如下：
    1. \[a,1\] 拓展到\[a,5\]，其中dim=1，表示在dim 1上进行拓展。
    2. len\(viewshape\)=2并且viewshape\[dim\]=1

3. 关于valid\_shape的说明：

    在动态图场景中，假设Tensor input \[a,1\] 扩展到 \[a,5\]，并设置ViewShape为 \[a,2\]，框架会通过pypto.loop循环生成 \[a,2\] 分块，并按偏移量拼接。此时若未传入valid\_shape，代码将默认生成全 \[a,2\] 的Tensor（如pypto.expand\_clone\(input, \[a,2\]\)）。

    然而，当总尺寸 \[a,5\] 无法被分块尺寸 \[a,2\] 整除时，尾块的有效形状（如 \[a,1\]）无法由框架自动推导。例如，最后一列可能仅包含1个元素，而非完整的 \[a,2\] 分块。此时必须通过valid\_shape明确指定尾块的实际有效Shape，如下：

    pypto.expand\_clone\(input, \[a,2\], valid\_shape = \[a, pypto.min\(2, 5 - 2 \* b\_idx\),\)

    其中b\_idx表示循环索引。

4. tileshape的维度与result维度相同，用于切分result。
5. tileshape的大小形状无额外约束，只需保证不超过ub size。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
# static graph
a = pypto.tensor([1,8], pypto.DT_INT32)
out1 = pypto.expand_clone(a, [4,8])
# dynamic graph
out2 = pypto.expand_clone(a, [4,8], valid_shape = [pypto.symbolic_scalar(4), pypto.symbolic_scalar(8)])
```

结果示例如下：

```python
输入数据a:     [[1, 2, 3, 4, 5, 6, 7, 8]]
输出数据out1:  [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]]
输出数据out2:  [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]]
```
