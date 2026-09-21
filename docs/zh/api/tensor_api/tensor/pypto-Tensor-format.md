# pypto.Tensor.format

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

获取Tensor的格式。

## 函数原型

```python
format(self) -> TileOpFormat
```

## 参数说明

无

## 返回值说明

TileOpFormat：返回Tensor的格式。

## 约束说明

这是一个只读属性。

## 调用示例

```python
t = pypto.tensor((4, 4), pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND)
print(t.format)
```

结果示例如下：

```text
输出：TileOpFormat.TILEOP_ND
```
