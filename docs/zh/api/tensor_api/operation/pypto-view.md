# pypto.view

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

从输入Tensor中取出部分视图，用于后续计算。

## 函数原型

```python
view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, dtype: DataType = None,
) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。<br>支持的数据类型为：PyPto支持的数据类型<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| shape       | 输入      | 获取出视图的大小，需要和input的维度数量一致。<br>Shape Size不大于2147483647（即INT32_MAX），**shape仅支持List [int] 类型，不支持SymbolicScalar类型。** |
| offsets     | 输入      | 获取视图时每个维度相对于input的偏移。<br>需要保证offsets小于input的Shape |
| valid_shape | 输入      | 取出视图块的有效数据大小。<br>需要保证valid_shape小于input的Shape；在类似page_attention场景下，当输入的kv_cache等张量包含无效数据时，无法正确推导输出的validshape，需要手动传入； |
| dtype       | 输入      | 返回值的数据类型，允许将输入数据解读为不同数据类型 |

`valid_shape` 可以配合 `pypto.experimental.assume_divisible` 使用。当某个动态标量已知可被 tile shape 整除时，可通过 `assume_divisible` 声明该事实，帮助编译器消除该轴逐 tile 动态 valid shape，使能 dualdst 等依赖静态 valid shape 的优化。

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape为参数shape指定大小，若指定了valid\_shape，则真实大小为valid\_shape。若指定dtype，则会将输入按照dtype进行读取。

## 约束说明

- **需要valid_shape时必须用pypto.view**：当需要指定`valid_shape`（动态有效数据大小）时，不能使用`[]`切片语法，必须使用显式的`pypto.view`接口
- 输入张量input和输入shape的维度数量需要一致。
- **view创建后与源input相互独立**：view创建后即成为独立的数据拷贝，对view的读写（含view[:] = ...）只作用于view自身，不会写回源input，也不会感知源input的后续修改。若需要将数据写回源Tensor（例如在循环中向persistent buffer分片累积写入的场景），请使用pypto.assemble(value, offsets, dest)。

## 调用示例

- 基本使用方法

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pypto.view(x, shape, offsets)
    ```

    结果示例如下：

    ```python
    输入数据x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4]]
    输出数据y: [[3 3 4 4],
                [3 3 4 4],
                [3 3 4 4],
                [3 3 4 4]]
    ```

- 增加valid\_shape

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    y = pypto.view(x, shape, offsets, valid_shape)
    ```

    结果示例如下：

    ```python
    输入数据x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 5 5 6 6],
                [1 1 2 2 5 5 6 6]]
    输出数据y: [[5 5 6 6],
                [5 5 6 6],
                [0 0 0 0],
                [0 0 0 0]]
    ```

- 指定dtype

    ```python
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.view(x, dtype=pypto.DT_INT8)
    ```

    结果如下：

    ```python
    输入数据x:
    [[0.9405094  0.20237109],
     [0.99819463 0.13246714]]

    输出数据y:
    [[  57  -59  112   63   94   58   79   62],
     [ -81 -119  127   63  119  -91    7   62]]

    ```
