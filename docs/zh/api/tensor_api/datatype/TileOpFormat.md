# TileOpFormat

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

TileOpFormat定义了Tensor的瓦片操作格式，用于优化不同计算模式下的内存访问和计算效率。主要区分密集计算和稀疏计算的瓦片格式。

## 原型定义

```python
class TileOpFormat(enum.Enum):
     TILEOP_ND = ...  # N维Tensor，支持标准的多维数组操作
     TILEOP_NZ = ...  # 同FRACTAL_NZ/NZ是对一个Tensor最低两维（一个Tensor的所有维度，右侧为低维，左侧为高维）进行填充（pad）、拆分（reshape）和转置（transpose）操作后得到的格式
```
