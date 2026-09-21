# vf.reg_tensor

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

向量寄存器，VF计算的基本数据容器。用于存储从UB Tile加载的数据、VF运算的中间结果和最终输出。

## 原型定义

```python
reg_tensor(dtype: DType) -> reg_tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dtype | 输入 | 寄存器存储的数据类型，决定寄存器覆盖的元素个数。 <br>- vf.reg_tensor不能直接调用，由编译器在赋值形式中自动声明。<br>- 寄存器在@pypto_pro.language.vector_function函数内创建和使用，函数结束后自动释放。<br>- 创建寄存器后必须通过vf.load_align或vf.full初始化数据，否则内容未定义。<br>- RegTensor寄存器数量上限为32。超出限制上限的寄存器数据会写入预留的8K UB内存中，可能会引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若寄存器与预留内存均存在可用空间，将优先复用寄存器。 |

## 约束说明

- 数据类型约束：

  RegTensor支持的数据类型由dtype参数决定，不同dtype对应不同的元素个数（寄存器总大小固定为256字节）：

  | dtype | 元素宽度 | 元素个数 |
  |---|---|---|
  | DT_INT8 / DT_UINT8 / DT_HF8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 | 8 bit | 256 |
  | DT_FP4 / DT_FP4E2M1 / DT_FP4E1M2 | 4 bit（b8 打包存储，2 元素/字节） | 256（逻辑 512） |
  | DT_INT16 / DT_UINT16 / DT_FP16 / DT_BF16 | 16 bit | 128 |
  | DT_INT32 / DT_UINT32 / DT_FP32 | 32 bit | 64 |
  | DT_INT64 / DT_UINT64 | 64 bit | 32 |

  > **FP8/FP4 说明**：FP8 类型（DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8）为 8 位浮点存储类型，仅支持数据搬运（load_align/store_align）、数据填充（full）和类型转换（astype），不支持直接参与算术运算。FP4 类型（DT_FP4E2M1、DT_FP4E1M2、DT_FP4）为 4 位浮点存储类型，两个元素打包在一个字节中（b8 存储），同样仅支持搬运、填充和类型转换。使用时需通过vf.astype转换为 FP32/BF16/FP16 进行计算。

- vf.reg_tensor为类型声明，不产生返回值。寄存器由赋值形式自动声明（如reg = vf.load_align(...)或reg = vf.add(...)）。

## 返回值说明

返回reg_tensor类型。

## 关键特性

### DT_FP16类型双寄存器模式存储结构

下图为DT_FP16在单寄存器模式和双寄存器模式场景下RegTensor存储情况：

**图1**RegTensor搬运DT_FP16

![](../../figures/reg_tensor_move_complex32.jpg "RegTensor搬运DT_FP16")

DT_FP16为16位浮点类型。在双寄存器模式场景下，从UB中以DIST_DINTLV_B16双搬入模式读取2*VL数据量，将数据交错搬运，偶数索引的元素存入reg[0]，奇数索引的元素存入reg[1]，数据类型为DT_UINT16。两个RegTensor存储512B的数据量，reg[0]和reg[1]各存128个DT_FP16元素。

### DT_FP32类型双寄存器模式存储结构

下图为DT_FP32在单寄存器模式和双寄存器模式场景下RegTensor存储情况：

**图2**RegTensor搬运DT_FP32

![](../../figures/reg_tensor_move_complex64.jpg "RegTensor搬运DT_FP32")

DT_FP32为32位浮点类型。在双寄存器模式场景下，从UB中以DIST_DINTLV_B32双搬入模式读取2*VL数据量，将数据交错搬运，偶数索引的元素存入reg[0]，奇数索引的元素存入reg[1]，数据类型为DT_UINT32。两个RegTensor存储512B的数据量，reg[0]和reg[1]各存64个DT_FP32元素。

### DT_INT64/DT_UINT64类型双寄存器模式存储结构

下图为b64（DT_INT64、DT_UINT64）在单寄存器模式和双寄存器模式场景下RegTensor存储情况：

**图3**RegTensor搬运b64

![](../../figures/reg_tensor_move_b64.jpg "RegTensor搬运b64")

在单寄存器模式场景下，从UB中以DIST_NORM模式搬运VL数据量。

在双寄存器模式场景下，从UB中以DIST_DINTLV_B32双搬入模式读取2*VL数据量，将b64数据交错搬运，偶数索引（低位）的元素存入reg[0]，奇数索引（高位）的元素存入reg[1]，数据类型为DT_UINT32。两个RegTensor存储512B的数据量，reg[0]存的是64个b64的前32位（低位），reg[1]存的是64个b64的后32位（高位）。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_b = vf.load_align(src_tile, 0)
    reg_out = vf.add(reg_a, reg_b, preg)
```
