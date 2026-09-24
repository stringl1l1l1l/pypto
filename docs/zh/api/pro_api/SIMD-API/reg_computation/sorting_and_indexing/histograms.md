# vf.histograms

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

对直方图数据进行统计，在目的操作数dst的基础数据上加上源操作数src数据的统计结果，包括数据的频率统计和累计统计。

### 频率统计

如下图所示，在低位模式（BIN0）下，dst用于统计src中index为[0-127]范围内（前半部分）各个值的出现频率；而在高位模式（BIN1）下，dst则统计[128-255]范围内（后半部分）的频率。dst中第n个元素表示src中数值n的出现次数，并在原始dst数据的基础上进行累加。

**图1** histograms频率统计

![](../../../figures/histograms_frequency_stats.jpg)

### 累计统计

如下图所示，在低位模式（BIN0）下，目的寄存器dst会统计源寄存器src中值落在低位区间[0-127]的数据分布情况；在高位模式（BIN1）下，目的寄存器dst则会统计src中值落在高位区间[128-255]的数据分布情况。在dst中，第n个元素表示src中从0到n的所有数值在对应区间中出现的总频率。最终，统计结果会在目的寄存器原始数据的基础上进行累加。

**图2** histograms累计统计

![](../../../figures/histograms_cumulative_stats.jpg)

## 函数原型

```python
histograms(src, preg, bin_type: Optional[BinType] = None, hist_type: Optional[HistType] = None)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，数据类型为DT_UINT8，待统计的数据。取值范围为0~255。<br>- mask位为0时，源操作数src对应位置的数值不参与统计，dst对应位置的值为原有值（对该位置src不存在的值进行统计）。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)，指定参与统计的元素范围。mask位为1时该元素参与统计，为0时不参与统计。dtype需为DT_UINT8（b8粒度）。 |
| bin_type | 输入 | 可选，分桶类型，决定统计的数据值范围，对应[BinType](../basic_data_structures/BinType.md)类型。<br>- pypto_pro.language.BinType.BIN0（默认）：低位模式，统计src中值在[0, 127]范围内的出现频率/累计频率，dst[n]对应数值n的统计结果。<br>- pypto_pro.language.BinType.BIN1：高位模式，统计src中值在[128, 255]范围内的出现频率/累计频率，统计时数值减去128映射到dst的对应位置，dst[n]对应数值(n+128)的统计结果。 |
| hist_type | 输入 | 可选，统计模式，决定计数方式，对应[HistType](../basic_data_structures/HistType.md)类型。<br>- pypto_pro.language.HistType.ACCUMULATE（默认，累计统计）：dst的第n个元素表示src中从0到n的所有数值在对应区间中出现的总频率，统计结果在dst原始数据基础上累加。<br>- pypto_pro.language.HistType.FREQUENCY（频率统计）：dst的第n个元素表示src中数值n的出现次数，统计结果在dst原始数据基础上累加。 |

## 约束说明

无。

## 返回值说明

返回dst目标[reg_tensor](../basic_data_structures/reg_tensor.md)，数据类型为DT_UINT16，存放分桶计数结果。VL(寄存器位宽)为256Byte时，dst可存储128个uint16元素。<br>- hist_type=ACCUMULATE时为原地累加：dst寄存器既被读又被写，首次调用前必须通过vf.full(0, ...)将dst初始化为零，后续dst = vf.histograms(...)复用同一寄存器继续累加。<br>- dst数据类型为DT_UINT16，最大值为65535，使用时需注意累加溢出问题。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    vreg = vf.load_align(src_tile, 0, dtype=pl.DT_UINT8)
    dst_reg = vf.full(0, preg_b16, dtype=pl.DT_UINT16)
    dst_reg = vf.histograms(vreg, preg_b8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE)
    vf.store_align(dst_tile, dst_reg, preg_b16)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
):
    tf_src = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    tf_dst = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_src, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_dst, addrs=0x100, mutex_ids=[1])
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
    a = torch.randint(0, 256, [1, 256], device=device, dtype=torch.uint8)
    out = torch.empty([1, 128], device=device, dtype=torch.int16)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    src_np = a.cpu().numpy().flatten()
    expected = torch.zeros(128, dtype=torch.int32, device=device)
    for v in src_np:
        if v <= 127:
            expected[v:] += 1
    torch.testing.assert_close(out[0].to(torch.int32), expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
