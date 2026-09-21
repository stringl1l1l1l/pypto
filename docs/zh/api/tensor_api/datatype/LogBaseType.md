# LogBaseType

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

LogBaseType定义了对数运算的底数类型，用于指定对数函数的计算底数，支持常用的自然对数、以2为底和以10为底的对数运算。

## 原型定义

```python
class LogBaseType(enum.Enum):
     LOG_E = ...   # 自然对数底数，以约2.718为底的对数运算
     LOG_2 = ...   # 以2为底的对数运算，常用于信息论和计算机科学
     LOG_10 = ...  # 以10为底的对数运算，常用于科学计算和工程应用
```
