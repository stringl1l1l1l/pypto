# vf.mem_bar

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

内存屏障，对src类内存操作与其后的dst类内存操作施加顺序保证，确保屏障前的操作对屏障后的操作可见。

如下图所示，目的流水线将等待源流水线上所有指令完成才进行执行。读写场景下，当读指令使用的寄存器和写指令使用的寄存器相同时，可以触发寄存器保序，指令将会按照代码顺序执行，不需要插入同步指令；而当使用的寄存器不同时，如果要确保读写指令顺序执行，则需要插入同步指令，写写场景同理。

**图1**流水线等待示意图

![流水线等待示意图](../../../figures/pipeline_wait.jpg)

通过mode选择src→dst的类型组合，共支持12种合法组合（V*表示矢量，*_LD、*_ST、ST_*、LD_*表示标量，*_ALL表示该单元全量屏障）：

| mode | src → dst含义 |
|---|---|
| VST_VLD | 矢量store → 矢量load（默认，RAW） |
| VLD_VST | 矢量load → 矢量store（WAR） |
| VST_VST | 矢量store → 矢量store（WAW） |
| VST_LD | 矢量store → 标量load |
| VST_ST | 矢量store → 标量store |
| VLD_ST | 矢量load → 标量store |
| ST_VLD | 标量store → 矢量load |
| ST_VST | 标量store → 矢量store |
| LD_VST | 标量load → 矢量store |
| VV_ALL | 全部矢量 ↔ 全部矢量 |
| VS_ALL | 全部矢量 ↔ 全部标量 |
| SV_ALL | 全部标量 ↔ 全部矢量 |

## 函数原型

```python
mem_bar(mode: Optional[MemBarMode] = None)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| mode | 输入 | 可选，屏障模式，对应[MemBarMode](../types/MemBarMode.md)类型，pypto_pro.language.MemBarMode枚举（见上表12种组合）。默认pypto_pro.language.MemBarMode.VST_VLD。mode只能取上表12种合法组合之一。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(in_a, t_f0):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(in_a, 0)
    reg_t = vf.add(reg_a, reg_a, preg)
    vf.store_align(t_f0, reg_t, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)
    reg_r = vf.load_align(t_f0, 0)
    vf.mem_bar(mode=pl.MemBarMode.VV_ALL)
    vf.store_align(t_f0, reg_r, preg)

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
    torch.testing.assert_close(out, a + a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
