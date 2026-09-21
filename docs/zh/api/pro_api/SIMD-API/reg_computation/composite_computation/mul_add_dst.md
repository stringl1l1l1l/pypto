# vf.mul_add_dst

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

该接口根据mask对src0、src1和dst逐元素执行乘加融合运算（FMA），将src0与src1相乘的积加上dst的当前值，结果写回dst。计算公式如下：

$$
dst_i = src0_i \times src1_i + dst_i
$$

由于乘法和加法在单条指令内完成，中间乘积不会因寄存器宽度限制而被截断或舍入，因此精度高于先调用vf.mul再调用vf.add的分步写法。

dst寄存器既被读取（作为加数）又被写入（存储结果），调用前必须预初始化为加数值。

## 函数原型

```python
mul_add_dst(src0, src1, preg, mode: Optional[MergeMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src0 | 输入 | 源操作数0，[reg_tensor](../reg_tensor.md)，源操作数src0、src1与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_FP16、DT_BF16、DT_FP32。 |
| src1 | 输入 | 源操作数1，[reg_tensor](../reg_tensor.md)，数据类型与src0一致。 |
| preg | 输入 | [mask_reg](../mask_reg.md)。 |
| mode | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.ZEROING（默认），preg未筛选的元素在dst中置0。<br>- pypto_pro.language.MergeMode.MERGING当前不支持。 |

## 约束说明

无。

## 返回值说明

返回dst目标/累加操作数，[reg_tensor](../reg_tensor.md)，支持的数据类型和src0中的说明一致。调用前需预初始化为加数值。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a_tile, src_b_tile, dst_tile, out_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a_tile, 0)
    reg_b = vf.load_align(src_b_tile, 0)
    reg_out = vf.load_align(dst_tile, 0)
    reg_out = vf.mul_add_dst(reg_a, reg_b, preg)
    vf.store_align(out_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    in_c_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    in_c = in_c_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x300, mutex_ids=[3])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.load(in_c, c, [0, 0])
        example_vf(in_a, in_b, in_c, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    c = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, c, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a * b + c, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
