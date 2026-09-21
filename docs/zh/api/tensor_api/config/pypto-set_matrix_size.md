# pypto.set\_matrix\_size

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

将NZ的输入Tensor在经过reshape后，传入matmul/scaled_mm使用计算时，需要传入参与计算Tensor在reshape前的m，k，n维度值，使matmul/scaled_mm获取到原始m，k，n维度的形状值。

## 函数原型

```python
set_matrix_size(size: List[int])-> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                            |
|---------|-----------|-------------------------------|
| size    | 输入      | 输入Tensor的m，k，n维度的形状值 |

## 返回值说明

void

## 约束说明

1、NZ输入的Tensor，在经过reshape后，调用matmul/scaled_mm计算时，需设置该参数。

2、调用matmul/scaled_mm的输入是3维/4维的NZ格式Tensor，需设置该参数。

## 调用示例

```python
a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
pypto.set_matrix_size([32, 64, 16]) #对应输入的Tensor的m，k，n维度的形状值
out = pypto.matmul(a, b, pypto.DT_FP32)
```
