# pypto_pro.language.system.bar_mte1

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

在MTE1流水中执行屏障同步，等待MTE1流水中此前下发的操作完成。

## 函数原型

```python
pypto_pro.language.system.bar_mte1() -> None
```

## 参数说明

无。

## 约束说明

- 仅支持在Cube区段中调用。
- 仅等待当前AI Core内MTE1流水中此前下发的操作，不执行跨核同步。

## 返回值说明

无。

## 调用示例

```python
with pl.section_cube():
    # ... MTE1流水操作
    pl.system.bar_mte1()
    # ... 后续操作
```
