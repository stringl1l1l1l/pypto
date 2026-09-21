# pypto_pro.language.FillPadMode

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

fillpad接口使用的填充模式枚举，用于指定源Tile与目标Tile的地址和形状关系。

## 原型定义

```python
class FillPadMode(enum.Enum):
    NORMAL = 0   # 源Tile与目标Tile的形状相同、地址不同。
    EXPAND = 1   # 目标Tile的各维形状不小于源Tile，并填充扩展区域。
    INPLACE = 2  # 源Tile与目标Tile的形状相同，并共享同一地址。
```
