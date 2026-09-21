# pypto.get\_verify\_options

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

获取当前已配置的精度调试Verify特性选项。

## 函数原型

```python
get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## 参数说明

无。

## 返回值说明

返回一个字典，包含精度调试Verify特性的当前设定值。

## 约束说明

## 调用示例

```python
pypto.get_verify_options()
```
