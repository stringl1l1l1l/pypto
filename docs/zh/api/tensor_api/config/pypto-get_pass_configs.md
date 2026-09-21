# pypto.get\_pass\_configs

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

获取指定Pass的全部配置信息。

## 函数原型

```python
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## 参数说明

| 参数名     | 输入/输出 | 说明                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| strategy   | 输入      | Pass策略名称，如"PVC2_OOO"。 |
| identifier | 输入      | Pass名称，如"ExpandFunction"。 |

## 返回值说明

PassConfigs对象，该对象含有以下属性，属性只读：

| 属性               | 说明                                                                 |
|--------------------|----------------------------------------------------------------------|
| printGraph         | dump计算图ir。 |
| dumpGraph          | dump计算图。 |
| dumpPassTimeCost   | dump Pass耗时。 |
| preCheck           | Pass执行前进行校验。 |
| postCheck          | Pass执行后进行校验。 |
| disablePass        | 不执行当前Pass。 |
| healthCheck        | 执行健康检查并生成报告。 |

## 约束说明

无

## 调用示例

```python
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```
