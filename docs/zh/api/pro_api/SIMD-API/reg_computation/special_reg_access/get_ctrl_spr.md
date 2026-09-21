# pypto_pro.language.get_ctrl_spr

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

读取CTRL特殊寄存器中指定比特区间的值。返回该比特区间的当前值。

## 函数原型

```python
get_ctrl_spr(start_bit: int, end_bit: int) -> int
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| start_bit | 输入 | 读取的特殊寄存器起始比特位（0-63），编译期常量。 |
| end_bit | 输入 | 读取的特殊寄存器结束比特位（0-63），编译期常量。 |

## 约束说明

- 读取范围不受可写比特位限制，可读取任意比特区间。

## 返回值说明

返回int类型，为CTRL寄存器中[start_bit, end_bit]比特区间的值。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    global_mode = pl.get_ctrl_spr(60, 60)
    float_sat = pl.get_ctrl_spr(48, 48)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.store(out, in_a, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, 1](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
