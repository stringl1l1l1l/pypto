# pypto.index\_add\_ub

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

pypto.index_add__ub的non-inplace版本，可参考[pypto.index_add__ub](pypto-index_add__ub.md)。

## 函数原型

```python
index_add_ub(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## 参数说明

请参考[pypto.index_add__ub](pypto-index_add__ub.md)的参数说明。

## 返回值说明

请参考[pypto.index_add__ub](pypto-index_add__ub.md)的返回值说明。

## 约束说明

1. 请参考[pypto.index_add__ub](pypto-index_add__ub.md)的约束说明。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

请参考[pypto.index_add__ub](pypto-index_add__ub.md)的调用示例。
