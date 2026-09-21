# pypto.reset\_options

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

重新设置所有配置项为默认值，包含codegen\_options，host\_options，pass\_options，runtime\_options和verify\_options。

## 函数原型

```python
reset_options() -> None
```

## 参数说明

无。

## 返回值说明

None：无返回值。设置操作成功即生效。

## 约束说明

无。

## 调用示例

```python
pypto.reset_options()
```
