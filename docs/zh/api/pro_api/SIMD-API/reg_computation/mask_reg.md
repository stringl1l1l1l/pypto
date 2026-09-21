# vf.mask_reg

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

掩码寄存器（mask_reg），VF计算的元素级有效性控制容器。用于控制VF算子中哪些元素参与运算。

mask_reg总位宽固定为256 bit，其粒度由关联的dtype参数决定。VF算子执行时，根据mask_reg中每个数据元素对应的比特位决定该元素是否参与运算：

- **比特位为1（有效）**：该元素参与运算，结果写入目的寄存器对应位置。
- **比特位为0（无效）**：该元素不参与运算，目的寄存器对应位置置零（vf.full等少数算子支持通过mode参数选择保留原值）。

## 原型定义

```python
mask_reg(dtype: DType) -> mask_reg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dtype | 输入 | 掩码粒度对应的数据类型，决定每多少bit对应一个数据元素。mask_reg总位宽固定为256 bit。<br>- vf.mask_reg不能直接调用，由编译器在赋值形式中自动声明。<br>- mask_reg在@pypto_pro.language.vector_function函数内创建和使用，函数结束后自动释放。<br>- mask_reg的dtype一般与配套的vf.reg_tensor的dtype一致，不一致的情况下需要自行判断结果行为。<br>- MaskReg寄存器数量上限为16。编译器会自动复用生命周期结束的寄存器和预留内存，若寄存器与预留内存均存在可用空间，将优先复用寄存器。 |

## 约束说明

- 数据类型约束：

  | dtype | 元素位宽 | 元素个数 | 每元素掩码位数 | 总掩码位数 |
  |---|---|---|---|---|
  | DT_INT8 / DT_UINT8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 / DT_HF8 / DT_FP4E2M1 / DT_FP4E1M2 | 8 bit | 256 | 1 bit（b8粒度） | 256 bit |
  | DT_FP16 / DT_UINT16 / DT_BF16 | 16 bit | 128 | 2 bit（b16粒度） | 256 bit |
  | DT_FP32 / DT_INT32 / DT_UINT32 | 32 bit | 64 | 4 bit（b32粒度） | 256 bit |
  | DT_INT64 / DT_UINT64 | 64 bit | 32 | 8 bit（b64粒度） | 256 bit |

- vf.mask_reg不能直接调用，由编译器在赋值形式中自动声明（如preg = vf.create_mask(...)或preg = vf.eq(...)）。

## 返回值说明

返回mask_reg类型。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    reg_out = vf.add(reg, reg, preg)
```
