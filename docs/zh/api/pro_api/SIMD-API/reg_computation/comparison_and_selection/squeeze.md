# vf.squeeze

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

将传入的src中被preg选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0，如下图所示。

$$dstReg_j = srcReg_{idx_j}, \quad j \in \{0, 1, \ldots, count\_active - 1\}$$

**图1** vf.squeeze计算示意图

![](../../../figures/squeeze_calculation.jpg)

特别地，当gather_mode取值为"pypto_pro.language.SqueezeMode.STORE_REG"时，vf.squeeze会将有效元素的总字节数存入AR特殊寄存器。此时配合使用连续非对齐搬出接口（无需显式传入偏移量），vf.store_unalign会自动从AR寄存器读取有效字节数作为地址偏移。

## 函数原型

```python
squeeze(src, preg, gather_mode: Optional[SqueezeMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../reg_tensor.md)，支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_INT32、DT_UINT32、DT_FP32。 |
| preg | 输入 | [mask_reg](../mask_reg.md)，指定哪些元素参与压缩。 |
| gather_mode | 输入 | 可选，收集模式："NO_STORE_REG"（不存入AR寄存器，默认）/ "STORE_REG"（有效元素总字节数存入AR寄存器）。<br>- 当gather_mode取值为"STORE_REG"时，由于硬件约束，vf.store_unalign指令和vf.squeeze指令必须交替使用。<br>- 当gather_mode取值为"NO_STORE_REG"时，不涉及AR寄存器，vf.squeeze和vf.store_unalign不强制交替。 |

## 约束说明

无。

## 返回值说明

返回dst目标[reg_tensor](../reg_tensor.md)，存放压缩后的元素，支持的数据类型和src中的说明一致。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_out = vf.squeeze(reg_src, preg, gather_mode=pl.SqueezeMode.NO_STORE_REG)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
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
    a = torch.randint(0, 100, [1, 64], device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    assert out.shape == torch.Size([1, 64])

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
