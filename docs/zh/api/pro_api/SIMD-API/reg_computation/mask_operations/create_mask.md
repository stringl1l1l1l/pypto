# vf.create_mask

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

创建mask_reg，指定参与后续VF运算的元素范围。

### mask_reg工作原理

mask_reg是VF运算中控制元素级有效性的专用寄存器。VF算子（如vf.add、vf.mul等）在执行时，会根据mask_reg中每个元素的对应比特位决定该元素是否参与运算：

- **比特位为1（有效）**：该元素参与运算，结果写入目的寄存器对应位置。
- **比特位为0（无效）**：该元素不参与运算，目的寄存器对应位置置零（vf.add、vf.max、vf.min、vf.full等少数算子支持通过mode参数选择保留原值）。

mask_reg的总位宽为256 bit，其粒度由dtype参数决定；每个数据元素对应的掩码位数随元素位宽变化。例如：

| dtype | 元素位宽 | 元素个数 | 每元素掩码位数 | 总掩码位数 |
|---|---|---|---|---|
| DT_INT8 / DT_UINT8 | 8 bit | 256 | 1 bit（8位宽粒度） | 256 bit |
| DT_FP16 / DT_UINT16 / DT_BF16 | 16 bit | 128 | 2 bit（16位宽粒度） | 256 bit |
| DT_FP32 / DT_INT32 / DT_UINT32 | 32 bit | 64 | 4 bit（32位宽粒度） | 256 bit |
| DT_INT64 / DT_UINT64 | 64 bit | 32 | 8 bit（64位宽粒度） | 256 bit |

> [!CAUTION]注意
> dtype参数决定的是掩码粒度（即mask_reg中每多少个bit对应一个数据元素），而非mask_reg本身的类型。mask_reg类型始终不变。

### mask_reg的典型使用场景

1. **全量运算**：pattern=ALL，所有元素参与运算（最常用）。
2. **尾块处理**：当数据长度不是寄存器宽度的整数倍时，用VL1~VL128限制最后一块的参与元素数。
3. **条件选择**：通过vf.eq、vf.gt等比较算子生成掩码，再用vf.select按掩码选择元素。
4. **交替处理**：用H、Q、M3、M4等模式对寄存器中的部分元素进行筛选运算。

以b8数据类型为例，不同MaskPattern模式下CreateMask接口的元素选取如下图所示：

**图1**b8数据类型下CreateMask接口不同MaskPattern模式下元素选取

![b8数据类型下CreateMask接口不同MaskPattern模式下元素选取](../../../figures/create_mask_b8_pattern_selection.jpg)

### astype精度转换中的mask_reg

不同数据类型下元素对应的mask位宽不一致，在astype进行类型转换时，mask_reg根据输入的源操作数进行有效元素筛选。

下图展示了mask_reg和RegLayout同时作用时16位宽和32位宽进行类型转换的过程：

**图2**astype 16位宽到32位宽类型转换过程

![[mask_reg](../basic_data_structures/mask_reg.md) 16位宽到32位宽类型转换过程](../../../figures/mask_reg_b16_to_b32_conversion.jpg)

**图3**astype 32位宽到16位宽类型转换过程

![mask_reg 32位宽到16位宽类型转换过程](../../../figures/mask_reg_b32_to_b16_conversion.jpg)

## 函数原型

```python
create_mask(pattern: Optional[MaskPattern] = None, dtype: Optional[DType] = None) -> preg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| pattern | 输入 | 可选，掩码模式，pattern参数决定mask_reg中哪些元素被设置为有效（1），哪些被设置为无效（0），对应[MaskPattern](../basic_data_structures/MaskPattern.md)类型。支持的模式见[约束说明](#约束说明)，默认pypto_pro.language.MaskPattern.ALL。 |
| dtype | 输入 | 可选，掩码对应的数据类型，决定掩码粒度（即每多少bit对应一个数据元素）。如pypto_pro.language.DT_FP32对应32位宽粒度（64元素 × 4 bit），全部对应关系请见[约束说明](#约束说明)。掩码寄存器总位宽固定为256 bit，默认pypto_pro.language.DT_FP32。 |

## 约束说明

- 数据类型约束：

  **表1** dtype对应数据类型掩码说明

  | dtype | 元素位宽 | 元素个数 | 每元素掩码位数 | 总掩码位数 |
  |---|---|---|---|---|
  | DT_INT8 / DT_UINT8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 / DT_HF8 / DT_FP4E2M1 / DT_FP4E1M2 | 8 bit | 256 | 1 bit（b8粒度） | 256 bit |
  | DT_FP16 / DT_UINT16 / DT_BF16 | 16 bit | 128 | 2 bit（b16粒度） | 256 bit |
  | DT_FP32 / DT_INT32 / DT_UINT32 | 32 bit | 64 | 4 bit（b32粒度） | 256 bit |
  | DT_INT64 / DT_UINT64 | 64 bit | 32 | 8 bit（b64粒度） | 256 bit |

- pattern参数说明：

  **表2** MaskPattern模式说明

  | 取值 | 含义 | 示意（以DT_FP32 / 64元素为例） |
  |---|---|---|
  | pypto_pro.language.MaskPattern.ALL | 所有元素有效 | 1111111111111111...1111（全1） |
  | pypto_pro.language.MaskPattern.ALLF | 所有元素无效 | 0000000000000000...0000（全0） |
  | pypto_pro.language.MaskPattern.VL1 | 最低1个元素有效 | 1000000000000000...0000 |
  | pypto_pro.language.MaskPattern.VL2 | 最低2个元素有效 | 1100000000000000...0000 |
  | pypto_pro.language.MaskPattern.VL4 | 最低4个元素有效 | 1111000000000000...0000 |
  | pypto_pro.language.MaskPattern.VL8 | 最低8个元素有效 | 1111111100000000...0000 |
  | pypto_pro.language.MaskPattern.VL16 | 最低16个元素有效 | 前16个1，其余0 |
  | pypto_pro.language.MaskPattern.VL32 | 最低32个元素有效 | 前32个1，其余0 |
  | pypto_pro.language.MaskPattern.VL64 | 最低64个元素有效 | 前64个1，其余0 |
  | pypto_pro.language.MaskPattern.VL128 | 最低128个元素有效 | 全部有效（仅8位宽/16位宽粒度下有意义） |
  | pypto_pro.language.MaskPattern.H | 最低一半元素有效 | 前32个1，后32个0（64元素时） |
  | pypto_pro.language.MaskPattern.Q | 最低四分之一元素有效 | 前16个1，后48个0（64元素时） |
  | pypto_pro.language.MaskPattern.M3 | 3的倍数位置有效 | 每第3个元素为1 |
  | pypto_pro.language.MaskPattern.M4 | 4的倍数位置有效 | 每第4个元素为1 |

## 返回值说明

返回preg目标[mask_reg](../basic_data_structures/mask_reg.md)。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    vf.store_align(dst_tile, reg, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
