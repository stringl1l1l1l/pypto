# pypto.scatter

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

将src的值写入input中。写入位置由index指定。3维计算公式如下，其他维度以此类推：

src为固定标量时：
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src & \text{if } dim = 2
\end{cases}
$$

src为Tensor时：
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 2
\end{cases}
$$

## 函数原型

```python
scatter(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

scatter\_的non-inplace版本，可参考[pypto.scatter\_](pypto-scatter_.md)

## 参数说明

请参考[pypto.scatter_](pypto-scatter_.md)的参数说明。

## 返回值说明

请参考[pypto.scatter_](pypto-scatter_.md)的返回值说明。

## 约束说明

1. 请参考[pypto.scatter_](pypto-scatter_.md)的约束说明。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

请参考[pypto.scatter_](pypto-scatter_.md)的调用示例。
