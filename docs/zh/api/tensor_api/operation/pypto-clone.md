# pypto.clone

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

将输入源数据拷贝一份并返回

## 函数原型

```python
clone(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的数据类型为：PyPTO支持的数据类型。<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回一个Shape、数据类型与输入一样的Tensor。

## 约束说明

1. 输入Tensor和输出Tensor类型应该相同。

## 调用示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.clone(x)
```

结果示例如下：

```python
input x: [[1, 2],
          [3, 4]]
output y: [[1, 2],
           [3, 4]]
```
