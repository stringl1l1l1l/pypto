# pypto.get\_pass\_default\_config

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

获取Pass的默认配置信息。

## 函数原型

```python
get_pass_default_config(key: PassConfigKey, default_value: bool) -> bool
```

## 参数说明

| 参数名          | 输入/输出 | 说明                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| key             | 输入      | PassConfigKey枚举。 <br> KEY_DUMP_GRAPH Dump Pass计算图。 |
| default_value   | 输入      | 若未找到指定key的配置值，则返回该默认值。 |

## 返回值说明

名为key的Pass默认配置值，若不存在返回default\_value。

## 约束说明

key仅能传入限定的枚举值。

## 调用示例

```python
pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```
