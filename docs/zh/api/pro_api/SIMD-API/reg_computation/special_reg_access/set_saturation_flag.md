# pypto_pro.language.set_saturation_flag

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

设置CTRL特殊寄存器中的饱和模式标志位。饱和模式控制vf.astype等类型转换指令在数据超出目标类型范围时的行为：

- **饱和模式（enable=True）**：超出目标类型范围的数据被钳位到目标类型的最大值或最小值。
- **不饱和模式（enable=False）**：超出目标类型范围的数据被截断为目标数据宽度的低位有效位。

模式配置与vf.astype的SaturateMode参数配合使用，具体生效规则请参考[Cast饱和模式配置表](../type_conversion/astype.md#约束说明)。

## 函数原型

```python
set_saturation_flag(mode: SaturationFlagMode, enable: bool) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| mode | 输入 | 饱和模式类别，对应[SaturationFlagMode](../basic_data_structures/SaturationFlagMode.md)枚举。<br>- pypto_pro.language.SaturationFlagMode.FLOAT：浮点数计算和浮点数精度转换（CTRL bit 48）<br>- pypto_pro.language.SaturationFlagMode.FLOAT8：浮点8计算（CTRL bit 50）<br>- pypto_pro.language.SaturationFlagMode.INT：整数计算（CTRL bit 53）<br>- pypto_pro.language.SaturationFlagMode.CAST：浮点转整数或整数转整数的精度转换（CTRL bit 59）<br> 设置后对后续所有VF计算指令生效，直到再次调用本接口修改。|
| enable | 输入 | 饱和模式使能位。True启用饱和模式，False禁用（不饱和模式）。 |

## 约束说明

- FLOAT/FLOAT8/CAST模式的极性为反转极性：bit=0表示饱和开启，bit=1表示饱和关闭。INT模式为正常极性：bit=1表示饱和开启，bit=0表示饱和关闭。

- 当vf.astype的saturate参数设置为pypto_pro.language.SaturateMode.ON或pypto_pro.language.SaturateMode.OFF时，为单指令模式（CTRL[60]=0），本接口设置的全局标志不生效。当需要全局饱和模式生效时，需确保CTRL[60]=1。

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
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=False)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
