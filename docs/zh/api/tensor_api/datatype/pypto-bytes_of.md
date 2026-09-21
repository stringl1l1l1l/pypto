# pypto.bytes\_of

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

返回某个数据类型所占的Byte的大小。

## 函数原型

```python
bytes_of(dtype: pypto.DataType) -> int
```

## 参数说明

| 参数名 | 输入/输出 | 说明                              |
|--------|-----------|-----------------------------------|
| dtype  | 输入      | 需要查看Byte大小的数据类型。     |

## 返回值说明

返回该数据类型所占的Byte的大小。

## 约束说明

无。

## 调用示例

```python
pypto.bytes_of(pypto.DT_FP32)
```

结果示例如下：

```python
输出: 4
```
