# pypto.Tensor.id

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

获取Tensor的唯一标识

## 函数原型

```python
id(self) -> int
```

## 参数说明

无

## 返回值说明

返回Tensor的唯一标识。

## 约束说明

这是一个只读属性。

## 调用示例

```python
t = pypto.tensor((4, 4), pypto.DT_FP32)
print(t.id)  # 输出Tensor的ID
```
