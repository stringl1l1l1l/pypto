# pypto.experimental.transposed_batchmatmul

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

该接口为定制接口，约束较多。不保证稳定性。

该算子执行转置批量矩阵乘法。具体操作为：

1. 将输入张量`tensor_a`从形状(M, B, K)转置为(B, M, K)。
2. 执行批量矩阵乘法，将转置后的`tensor_a` (B, M, K)与`tensor_b` (B, K, N)相乘，得到中间结果(B, M, N)。
3. 将中间结果转置回形状(M, B, N)作为最终输出。

## 函数原型

```python
transposed_batchmatmul(tensor_a: Tensor, tensor_b: Tensor, out_dtype: dtype) -> Tensor
```

## 参数说明

| 参数名    | 输入/输出 | 说明                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| tensor_a  | 输入      | 左侧输入张量。<br>支持的数据类型为：DT_FP16，DT_BF16。<br>不支持空Tensor，支持三维。<br>形状必须为(M, B, K)。 |
| tensor_b  | 输入      | 右侧输入张量。<br>支持的数据类型为：DT_FP16，DT_BF16。<br>不支持空Tensor，支持三维。<br>形状必须为(B, K, N)。 |
| out_dtype | 输入      | 输出张量的数据类型。<br>支持的数据类型为：DT_FP16，DT_BF16。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型由`out_dtype`指定，形状为(M, B, N)。

## 约束说明

本接口为试验特性，后续版本可能会存在变更，不支持应用于生产环境中。

## 调用示例

```python
import pypto

# 创建输入张量
a = pypto.tensor((16, 2, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((2, 32, 64), pypto.DT_FP16, "tensor_b")

# 调用算子
c = pypto.experimental.transposed_batchmatmul(a, b, pypto.DT_FP16)

# 输出张量c的形状为(16, 2, 64)
```
