# pypto_pro.language.system.bar_mte2

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

在MTE2流水中执行屏障同步，等待MTE2流水中此前下发的操作完成。

## 函数原型

```python
pypto_pro.language.system.bar_mte2() -> None
```

## 参数说明

无。

## 约束说明

- 支持在Cube区段或Vector区段中调用。
- 仅等待当前AI Core内MTE2流水中此前下发的操作，不执行跨核同步。
- 连续的MTE2搬入操作写入的片上地址存在重叠时，应在两次操作之间调用本接口，保证前一次搬入完成后再执行下一次搬入，否则可能产生错误数据。

## 返回值说明

无。

## 调用示例

```python
with pl.section_vector():
    # ... MTE2流水操作
    pl.system.bar_mte2()
    # ... 后续操作
```
