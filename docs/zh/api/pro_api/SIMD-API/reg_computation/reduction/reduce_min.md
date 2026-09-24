# vf.reduce_min

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

reg_tensor最小值归约：将源寄存器src中的所有有效元素（preg选中的元素）求最小值，结果写入目标寄存器的第一个元素dst[0]，第一个最小值所在索引写入dst[1]，其余元素置零。

归约求最小值的计算过程及索引保存方式如下图所示：

**图1** reduce_min归约索引示意图

![reg_reduce_index示意图](../../../figures/reg_reduce_index.jpg)

## 函数原型

```python
reduce_min(src, preg, datablock: bool = False, merge_mode: Optional[MergeMode] = None)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，源操作数src与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_INT16、DT_UINT16、DT_FP16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)。当所有元素均不参与计算时（mask为空），将该数据类型的最大值写入dst[0]。 |
| datablock | 输入 | 可选，决定接口工作模式，True时按datablock粒度归约（对应vcgmin指令），默认False。当datablock=True时，启用datablock粒度归约，每个datablock独立归约：32位宽（DT_INT32、DT_UINT32、DT_FP32）类型每16个元素为一个datablock，16位宽（DT_INT16、DT_UINT16、DT_FP16）类型每32个元素为一个datablock，各datablock分别求最小值并将结果依次写入dst的最低位。 |
| merge_mode | 输入 | 可选，对应[MergeMode](../basic_data_structures/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.ZEROING（默认），preg未筛选的元素在dst中置0。<br>- pypto_pro.language.MergeMode.MERGING当前不支持。 |

## 约束说明

- datablock=True时，支持的数据类型为：DT_INT16、DT_UINT16、DT_FP16、DT_INT32、DT_UINT32、DT_FP32。

## 返回值说明

返回dst目标[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型和src中的说明一致，归约结果写入第一个元素dst[0]，索引写入dst[1]。

- 当存在多个最小值时，将第一个最小值的索引保存在dst[1]中。

- 如果输入数据存在nan，将该数据类型的nan写入dst[0]，并将第一个nan的索引保存在dst[1]中。

- min(-0, +0) = -0。

## 调用示例

### 基本调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src0 = vf.load_align(src_tile, 0)
    min0 = vf.reduce_min(src0, preg_all)
    vf.store_align(dst_tile, min0, preg_all)

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
    torch.testing.assert_close(out[0, 0], torch.min(a), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### INT64数据类型示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.reduce_min(reg_a, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_int64(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
):
    tf = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int64(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int64():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.zeros([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[0, 0], torch.min(a), rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
