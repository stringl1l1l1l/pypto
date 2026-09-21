# pypto_pro.language.set_ctrl_spr

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

设置CTRL特殊寄存器中指定比特区间的值。CTRL寄存器控制矢量计算的多种全局模式，包括饱和模式、原子操作配置等。

## 函数原型

```python
set_ctrl_spr(start_bit: int, end_bit: int, value: int) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| start_bit | 输入 | 设置的特殊寄存器起始比特位（0-63），编译期常量。可配置bit位见[约束说明](#约束说明)。 |
| end_bit | 输入 | 设置的特殊寄存器结束比特位（0-63），编译期常量。 |
| value | 输入 | 写入特殊寄存器指定比特区间的值。 |

## 约束说明

- 可写的CTRL比特位：6-10（范围）、45、48、50、53、59、60（单比特）。其他比特位不可写。各比特位说明：

  | 比特位 | 名称 | 说明 |
  |---|---|---|
  | 6-10 | 原子配置 | 原子操作类型和模式，用于控制数据从L0C Buffer/Unified Buffer（UB）/L1 Buffer搬出到Global Memory时的原子操作。<br>- CTRL[8:6]，控制原子操作的数据类型：<br>&nbsp;&nbsp;- 3'b000：禁止原子操作（默认）；<br>&nbsp;&nbsp;- 3'b001：数据类型为DT_FP32；<br>&nbsp;&nbsp;- 3'b010：数据类型为DT_FP16；<br>&nbsp;&nbsp;- 3'b011：数据类型为DT_INT16；<br>&nbsp;&nbsp;- 3'b100：数据类型为DT_INT32；<br>&nbsp;&nbsp;- 3'b101：数据类型为DT_INT8；<br>&nbsp;&nbsp;- 3'b110：数据类型为DT_BF16。<br>- CTRL[10:9]，控制原子操作类型，仅在CTRL[8:6]配置原子操作时有效：<br>&nbsp;&nbsp;- 2'b00：ADD，累加原子操作（默认）；<br>&nbsp;&nbsp;- 2'b01：MAX，取最大值原子操作；<br>&nbsp;&nbsp;- 2'b10：MIN，取最小值原子操作。 |
  | 45 | matmul配置 | 用于控制matmul运算时寄存器的格式。<br>- 1'b0：按照原子操作数类型进行转换（默认）。<br>- 1'b1：当复数数据均为DT_FP8E4M3FN时，转换为DT_HF8再进行乘法运算；其他数据按照原子操作数类型进行转换。 |
  | 48 | 浮点数饱和 | 用于控制浮点数据类型转换时的饱和模式（仅在CTRL[60]配置为全局比特模式时有效）。<br>- 1'b0：饱和模式，inf会被转换为最大MAX，NaN会被转换为0（默认）。<br>- 1'b1：非饱和模式，inf/NaN按照原样处理。<br>该控制位仅支持以下数据类型：原子操作时支持DT_FP16；浮点数据类型转换时支持DT_HF8、DT_FP8E8M0、DT_FP8E5M2、DT_FP8E4M3FN、DT_FP16、DT_BF16。 |
  | 50 | 浮点数NaN处理 | 用于控制浮点数据类型转换时的NaN处理模式（仅在CTRL[48]配置为饱和模式时有效）。<br>- 1'b0：NaN将会被转换为0.0（默认）。<br>- 1'b1：NaN将会保持NaN。<br>该控制位仅支持以下数据类型：DT_FP8E8M0、DT_FP8E5M2、DT_FP8E4M3FN。 |
  | 53 | 整数饱和 | 用于控制整数相关指令的饱和模式。<br>- 1'b0：截断模式，计算值到目的操作数位宽截断，高位截断保留低位（默认）。<br>- 1'b1：饱和模式，计算值大于MAX时饱和。 |
  | 59 | astype饱和 | 用于控制浮点转换指令和整数转换指令的精度转换饱和模式（仅在CTRL[60]配置为全局比特模式时有效）。<br>- 1'b0：饱和模式，计算值大于最大MAX时饱和（默认）。<br>- 1'b1：截断模式，计算值到目的操作数位宽截断，高位截断保留低位。 |
  | 60 | 饱和模式全局配置 | 用于控制饱和模式的全局生效方式。<br>- 1'b0：单指令配置生效。<br>- 1'b1：全局配置生效（默认）。<br>该控制位可与vf.astype等精度转换API配合使用，也可与CTRL[48]、CTRL[59]配合使用。 |

- 设置后对后续所有VF计算指令生效，直到再次调用本接口或pypto_pro.language.reset_ctrl_spr修改。

## 返回值说明

无。

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
    reg_i16 = vf.astype(reg, preg, dtype=pl.DT_INT16, layout=pl.CastLayout.ZERO)
    reg_f32 = vf.astype(reg_i16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    pl.set_ctrl_spr(60, 60, 1)
    pl.set_ctrl_spr(59, 59, 0)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])
    pl.reset_ctrl_spr(59, 59)
    pl.reset_ctrl_spr(60, 60)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, 1](a, out)
    torch.npu.synchronize()
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
