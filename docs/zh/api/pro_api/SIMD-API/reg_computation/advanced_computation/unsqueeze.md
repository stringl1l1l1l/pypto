# vf.unsqueeze

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

以dst为操作对象，根据preg进行解压缩。

具体算法如下图所示，dst的首位为0，后续mask[i]对应mask值为1时，dst[i]的值为dst[i-1] + 1；mask[i]对应mask值为0时，dst[i]的值为dst[i-1]。mask最高位被忽略不参与统计。

$$dstReg_i = \begin{cases} 1 & \text{if } mask_i = 1 \\ 0 & \text{if } mask_i = 0 \end{cases}$$

**图1** unsqueeze示意图

![Unsqueeze示意图](../../../figures/unsqueeze_diagram.jpg)

mask_reg由vf.create_mask或vf.update_mask产生，作为mask_reg类型的参数直接传递给矢量计算API，控制哪些元素参与运算。

## 函数原型

```python
unsqueeze(preg) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| preg | 输入 | [mask_reg](../mask_reg.md)（由vf.create_mask或vf.update_mask产生）。 |

## 约束说明

无。

## 返回值说明

返回dst目标操作数，[reg_tensor](../reg_tensor.md)，存放扩展结果，支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    src = vf.load_align(src_tile, 0)
    dst = vf.unsqueeze(preg)
    vf.store_align(dst_tile, dst, preg)

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
