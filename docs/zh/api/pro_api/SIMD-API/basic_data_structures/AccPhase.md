# pypto_pro.language.AccPhase

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

开启硬件unitFlag机制（一种Mmad指令和Fixpipe指令细粒度的并行）的枚举。硬件给L0C Buffer中每一个512B内存块提供一个unit-flag标志位，用于表示该内存块是否可被读写，进而从硬件层面实现L0C Buffer内存的读写同步。
对于矩阵乘而言，当unit-flag标志位为0时，可写入L0C Buffer，否则阻塞写操作，直到标志位变为0。使用时，需要与L0C Buffer搬出接口（[pypto_pro.language.store](../memory_data_movement/store.md)、[pypto_pro.language.store_tile](../memory_data_movement/store_tile.md)或[pypto_pro.language.move](../memory_data_movement/move.md)）的[pypto_pro.language.STPhase](STPhase.md)配置搭配使用，否则可能出现卡死现象。

## 原型定义

```python
PYPTO_DECLARE_ENUM(AccPhase,
    Unspecified,  # 不启用unitFlag机制
    Partial,      # 仅检查unit-flag标志位，并根据其0/1状态控制写操作，指令执行后，不改变该内存块的unit-flag标志位。
    Final         # 检查unit-flag标志位，并根据其0/1状态控制写操作，指令执行后，将该内存块的unit-flag标志位置为1。
)
```
