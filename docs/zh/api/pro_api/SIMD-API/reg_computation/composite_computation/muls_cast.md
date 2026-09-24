# vf.muls_cast

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

该接口用于将源操作数src与标量scalar相乘，再按照layout将结果转换为DT_FP16类型，根据preg将计算结果写入目的操作数dst。计算公式如下：

$$
dst_i = cast\_round\_to\_f16(src_i \times scalar)
$$

## 函数原型

```python
muls_cast(src, scalar, preg, dtype: DType, layout: Optional[CastLayout] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型请参见[约束说明](#约束说明)。 |
| scalar | 输入 | 标量源操作数。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)。 |
| dtype | 输入 | 指定目标reg_tensor的数据类型，支持的数据类型请参见[约束说明](#约束说明)。由于乘法后进行类型转换（DT_FP32→DT_FP16），目标类型与源类型不同，必须显式指定，通常为pypto_pro.language.DT_FP16。 |
| layout | 输入 | 可选，结果放置半区：pypto_pro.language.CastLayout.ZERO（偶数半区，默认，PART_EVEN）或pypto_pro.language.CastLayout.ONE（奇数半区，PART_ODD），对应[CastLayout](../basic_data_structures/CastLayout.md)类型。计算按照CAST_ROUND模式舍入。 |

## 约束说明

- 数据类型约束：

  | src | scalar | dst |
  |---|---|---|
  | DT_FP32 | DT_FP32 | DT_FP16 |

- 不支持源操作数寄存器和目的操作数寄存器重叠。

## 返回值说明

返回dst目的操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型请参见[约束说明](#约束说明)。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src_tile, 0)
    reg_f16 = vf.muls_cast(reg_src, 2.0, preg, dtype=pl.DT_FP16)
    reg_out = vf.astype(reg_f16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_out, preg)

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
    expected = (a * 2.0).to(torch.float16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
