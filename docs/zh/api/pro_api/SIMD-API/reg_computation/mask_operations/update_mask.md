# vf.update_mask

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

从标量值更新mask_reg。vf.update_mask根据当前scalarValue的值生成对应长度的有效位掩码。以16位宽数据类型为例，掩码生成过程如下图所示：

**图1**update_mask 16位宽数据类型下基于scalarValue的掩码生成

![update_mask 16位宽数据类型下基于scalarValue的掩码生成](../../../figures/maskreg_b16_update_mask_gen.jpg)

## 函数原型

```python
update_mask(scalar, dtype: Optional[DType] = None) -> preg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| scalar | 输入 | 标量值，其比特位定义新的掩码模式。 |
| dtype | 输入 | 可选，掩码对应的数据类型，决定掩码宽度（默认pypto_pro.language.DT_FP32）。<br>- 本接口操作数为寄存器，不涉及地址对齐。<br>- 本接口不修改全局寄存器的值。 |

## 约束说明

- 数据类型约束：

  dtype参数决定掩码粒度（即每多少 bit 对应一个数据元素），掩码寄存器总位宽固定为 256 bit：

  | dtype | 元素位宽 | 元素个数 | 每元素掩码位数 | 总掩码位数 |
  |---|---|---|---|---|
  | DT_INT8 / DT_UINT8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 / DT_HF8 / DT_FP4E2M1 / DT_FP4E1M2 | 8 bit | 256 | 1 bit（b8 粒度） | 256 bit |
  | DT_FP16 / DT_UINT16 / DT_BF16 | 16 bit | 128 | 2 bit（b16 粒度） | 256 bit |
  | DT_FP32 / DT_INT32 / DT_UINT32 | 32 bit | 64 | 4 bit（b32 粒度） | 256 bit |
  | DT_INT64 / DT_UINT64 | 64 bit | 32 | 8 bit（b64 粒度） | 256 bit |

  > **注意**：FP8 类型（FP8E4M3FN/FP8E5M2/FP8E8M0/HF8）和 FP4 类型（FP4E2M1/FP4E1M2）均为 b8 存储，按 b8 粒度处理。掩码寄存器始终为[mask_reg](../basic_data_structures/mask_reg.md)类型。

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
    preg = vf.update_mask(0xFFFFFFFF, dtype=pl.DT_FP16)
    reg = vf.load_align(src_tile, 0)
    vf.store_align(dst_tile, reg, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
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
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
