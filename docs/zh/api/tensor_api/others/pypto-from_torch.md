# pypto.from\_torch

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

将一个torch.Tensor转换为pypto.Tensor。可显式指定转换后的pypto.Tensor的名称。可将转换后的pypto.Tensor的指定维度标记为动态维度，用于表示该维度在后续编译/运行阶段可变。

## 函数原型

```python
from_torch(tensor: torch.Tensor, name: str="", *, dynamic_axis: Optional[List[int]] = None,
           tensor_format: Optional[TileOpFormat] = None, dtype: Optional[DataType] = None) -> pypto.Tensor
```

## 参数说明

| 参数名         | 输入/输出 | 说明                                                                 |
|----------------|-----------|----------------------------------------------------------------------|
| tensor         | 输入      | 需要转换为pypto.Tensor的torch.Tensor对象。 |
| name           | 输入      | pypto.Tensor的名称。默认为空字符串，表示由from_torch自动为其命名。 |
| dynamic_axis   | 输入      | 要标记为动态的维度索引列表。默认为None，表示不标记任何维度。 |
| tensor_format  | 输入      | 要指定的pypto.TileOpFormat格式。为None时根据Tensor NPU Format自动推导。 |
| dtype      | 输入      | 要指定的pypto.DataType类型。为None时根据torch.Tensor的dtype自动推导。 |

## 返回值说明

返回转换后的pypto.Tensor。

## 约束说明

- 入参tensor类型必须为torch.Tensor或其子类。
- 入参tensor在指定内存格式的顺序下是连续的（tensor.is\_contiguous\(\) == True）。
- 入参tensor支持如下数据类型（dtype）：
    - torch.float16
    - torch.bfloat16
    - torch.float32
    - torch.float64
    - torch.int8
    - torch.uint8
    - torch.int16
    - torch.uint16
    - torch.int32
    - torch.uint32
    - torch.int64
    - torch.uint64
    - torch.bool

## 调用示例

```python
x= torch.randn(2, 3)
x_pto = pypto.from_torch(x)
print(x_pto.shape)
y = torch.randn(2, 3)
y_pto = pypto.from_torch(y, "y", dynamic_axis=[0])
print(y_pto.shape)
z = torch.randn(2, 3)
z_pto = pypto.from_torch(z, "z", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
print(z_pto.format)
k = torch.randn(2, 3)
k_pto = pypto.from_torch(k, "k", dtype=pypto.DataType.DT_HF8)
print(k_pto.dtype)
```

结果示例如下：

```python
[2, 3]
[SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_input_tensor,0)), 3]
TileOpFormat.TILEOP_NZ
DataType.DT_HF8
```
