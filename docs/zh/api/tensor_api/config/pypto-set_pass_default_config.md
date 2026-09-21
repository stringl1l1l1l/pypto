# pypto.set\_pass\_default\_config

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

修改Pass的默认配置信息。其主要功能是动态修改Pass的运行时行为配置，目前可配置dump计算图开关，方便分析和调试。

## 函数原型

```python
set_pass_default_config(key: PassConfigKey, value: bool)
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| key     | 输入      | PassConfigKey枚举 <br> KEY_DUMP_GRAPH Dump Pass计算图 |
| value   | 输入      | 配置值 |

## 返回值说明

无

## 约束说明

- 设置时机：必须在图编译开始前调用。
- 作用范围：配置信息是全局性的，会影响后续所有的编译过程。

## 调用示例

```python
pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```
