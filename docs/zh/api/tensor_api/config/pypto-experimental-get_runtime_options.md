# pypto.experimental.get\_runtime\_options

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

获取当前config中运行时的实验性配置信息。

## 函数原型

```python
get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## 参数说明

无。

## 返回值说明

返回dict，包含runtime的实验性配置项信息。

## 约束说明

1. 返回值为Dict类型，包含runtime的运行时配置项信息。
2. 不同配置项的类型可能不同：str、int、List[int]、List[str] 等。

## 调用示例

```python
pypto.experimental.get_runtime_options()
```
