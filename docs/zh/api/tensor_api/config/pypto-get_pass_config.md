# pypto.get\_pass\_config

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

获取指定Pass的配置信息。

## 函数原型

```python
get_pass_config(strategy: str, identifier: str, key: PassConfigKey, default_value: bool) -> bool
```

## 参数说明

| 参数名          | 输入/输出 | 说明                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| strategy        | 输入      | Pass策略名称，如"PVC2_OOO"。 |
| identifier      | 输入      | Pass名称，如"ExpandFunction"。 |
| key             | 输入      | PassConfigKey枚举。 <br> KEY_DUMP_GRAPH Dump Pass计算图。 |
| default_value   | 输入      | 若未找到指定key的配置值，则返回该默认值。 |

## 返回值说明

strategy策略下名称是identifier的Pass中key的配置值。

## 约束说明

key仅能传入限定的枚举值。

## 调用示例

```python
pypto.get_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
```
