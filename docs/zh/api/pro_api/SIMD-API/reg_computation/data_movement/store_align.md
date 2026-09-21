# vf.store_align

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

将reg_tensor或mask_reg数据对齐存储到Tile。支持reg_tensor和mask_reg两种寄存器类型：

### reg_tensor模式

reg_tensor模式支持**连续搬运模式**和**非连续搬运模式**。连续搬运模式又分为单搬出模式和双搬出模式：

- **reg_tensor单搬出模式**：将一个reg_tensor中的VL数据量搬出到Tile。
- **reg_tensor双搬出模式**（interleave）：将两个reg_tensor中的元素交错搬出到Tile，dst长度为2*VL。

**连续搬运模式**数据搬出时，可以通过dist关键字参数配置搬运的数据分布模式，能够实现压缩、只搬出第一个元素等功能。下图展示了部分分布模式的搬出示意：

**图1** vf.store_align连续对齐搬出分布模式图示

![](../../../figures/contiguous_aligned_store.jpg)

**非连续搬运模式**下，通过data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY启用DataBlock搬运模式，实现数据从reg_tensor非连续搬运至Tile。单条指令一次搬运8个DataBlock，block_stride参数表示相邻DataBlock间的间隔。该模式下preg参数位置改为传入控制有效元素的mask_reg，mask控制规则如下：

- 某个DataBlock在mask中对应的32bit有任意一位为1时，该DataBlock对应的数据会被搬出到Tile。
- 某个DataBlock在mask中对应的32bit全为0时，该DataBlock对应的数据不会被写出，Tile对应位置不更新，即使Tile越界也不会报错。

当同时指定post_update=True时，搬运后目标地址会按照repeat_stride参数自动更新：

- post_update=False时，实际搬运Tile起始地址为dstAddr，搬运后地址不更新。
- post_update=True时，实际搬运Tile起始地址为dstAddr，搬运后执行地址更新dstAddr += repeat_stride * 32B（单位为DataBlock，32字节），repeat_stride需要32字节对齐。

### mask_reg模式

当源操作数已通过vf.create_mask预声明为mask_reg时，后端自动分派mask_reg存储路径，将mask_reg中的数据搬出到Tile。

mask_reg源支持两种分布模式，通过dist关键字参数配置：

- pypto_pro.language.StoreDist.NORM（正常模式，搬运VL/8数据量）
- pypto_pro.language.StoreDist.PACK（压缩模式，每间隔1bit舍弃数据，将VL/8的数据压缩为VL/16搬出）

mask_reg模式同样支持三类搬运接口（普通搬运/PostUpdate/AddrReg）。

## 函数原型

```python
store_align(tile, src, preg, offset, dist=None, data_copy_mode=None, block_stride=None, repeat_stride=None, post_update=False)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile | 输出 | 目的操作数，Tile地址。地址需要32字节对齐。 |
| src | 输入 | 源操作数，[reg_tensor](../reg_tensor.md)或者[mask_reg](../mask_reg.md)类型。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。<br>- 当源为reg_tensor时，单一src时，为**reg_tensor单搬出模式**；输入src_even和src_odd两个src时，为**reg_tensor双搬出模式**。<br>- 当源已通过vf.create_mask预声明为mask_reg时，自动分派mask_reg存储路径，将mask_reg中的数据搬出到Tile。 |
| preg | 输入 | [mask_reg](../mask_reg.md)，指定写入的元素范围。**mask_reg模式**时无需传入。<br>- **连续搬运模式**下，mask中对应的bit为1时，该元素被写入Tile；为0时，该元素不被写入，Tile对应位置保持原值不变。**非连续搬运模式**（DataBlock模式下data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY），该位置改为传入控制有效元素的mask_reg，mask控制规则如下：某个DataBlock在mask中对应的32bit有任意一位为1时搬出，全为0时不写入且Tile对应位置不更新，即使Tile越界也不会报错。 |
| offset | 输入 | 可选，末尾位置参数（第4个），根据模式和post_update取值自动分派语义，单位为元素个数。在**非连续搬运模式**（data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY）下该参数位置为block_stride（见下）。<br>- **连续搬运模式 + post_update=False**：作为地址偏移量。<br>&nbsp;&nbsp;- **整数或[row, col]列表**：整数偏移在代码生成时转换为指针算术Tile + offset。传入[row, col]列表或元组，此时线性偏移为row * shape[1] + col，row单位为Tile的列数（即set_validshape[m, n]的n），col单位为元素个数，两者均支持表达式。<br>&nbsp;&nbsp;- **AddrReg**（由vf.create_addr_reg创建）：实际搬运Tile地址为Tile + AddrReg中存储的偏移量。每次迭代需先调用vf.create_addr_reg设定偏移量再调用搬运指令。<br>- **连续搬运模式 + post_update=True**：作为PostUpdate地址累进步长（元素数），搬运后目标地址自动更新为dstAddr += stride。默认0。64位宽数据类型（DT_INT64、DT_UINT64）自动翻倍。<br>- **mask_reg模式**（源为mask_reg时）：<br>&nbsp;&nbsp;- **整数**：仅在post_update=True时生效，作为地址更新步长；post_update=False时不支持整数offset。<br>&nbsp;&nbsp;- **AddrReg**（由vf.create_addr_reg创建）：实际搬运Tile地址为dstAddr + AddrReg中存储的偏移量。 |
| dist | 输入 | 可选，数据存储分布模式，对应[StoreDist](../types/StoreDist.md)类型，具体模式根据是**reg_tensor单搬出模式**、**reg_tensor双搬出模式**还是**mask_reg模式**请分别参见[约束说明](#约束说明)中各表。 |
| data_copy_mode | 输入 | 可选，数据拷贝模式，对应[DataCopyMode](../types/DataCopyMode.md)类型。仅在src为reg_tensor下有效，mask_reg源不支持此参数。取值：pypto_pro.language.DataCopyMode.NORM（默认，普通连续搬运）或pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY（非连续以DataBlock（32B）为单位进行搬运）。 |
| block_stride | 输入 | 可选，仅在data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY模式下有效，其他模式下传入会被忽略。表示相邻DataBlock间的间隔，单位：DataBlock（32字节）。可作位置参数（第4个）或关键字参数传入。 |
| repeat_stride | 输入 | 可选，仅在data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY模式且post_update=True时有效。表示重复搬运时的地址更新步长，单位：DataBlock（32字节），需要32字节对齐。post_update=True时，搬运后目标地址自动更新为dstAddr += repeat_stride * 32B。可作位置参数（第5个）或关键字参数传入。 |
| post_update | 输入 | 可选，True时搬运后目标地址自动累进，默认False。适用于循环内连续存储。 |

## 约束说明

- 各模式下dist参数说明：

  **表1** reg_tensor单搬出模式dist参数说明

  | dist取值 | 含义 | 对齐约束（Byte） |
  |---|---|---|
  | pypto_pro.language.StoreDist.NORM | 正常模式（通用），根据数据类型自动选择位宽粒度，搬运VL数据。64位宽数据类型DT_INT64、DT_UINT64只支持此模式。 | 32 |
  | pypto_pro.language.StoreDist.NORM_B8 | 按8位宽类型普通存储，搬运VL数据。 | 32 |
  | pypto_pro.language.StoreDist.NORM_B16 | 按16位宽类型普通存储，搬运VL数据。 | 32 |
  | pypto_pro.language.StoreDist.NORM_B32 | 按32位宽类型普通存储，搬运VL数据。 | 32 |
  | pypto_pro.language.StoreDist.FIRST_ELEMENT | 忽略mask，仅向dst搬出src第一个元素（通用），根据数据类型自动选择位宽粒度。 | 按dtype宽度 |
  | pypto_pro.language.StoreDist.FIRST_ELEMENT_B8 | 忽略mask，仅向dst搬出src第一个元素，数据类型为8位宽类型。 | 1 |
  | pypto_pro.language.StoreDist.FIRST_ELEMENT_B16 | 忽略mask，仅向dst搬出src第一个元素，数据类型为16位宽类型。 | 2 |
  | pypto_pro.language.StoreDist.FIRST_ELEMENT_B32 | 忽略mask，仅向dst搬出src第一个元素，数据类型为32位宽类型。 | 4 |
  | pypto_pro.language.StoreDist.PACK | 压缩模式（通用），根据mask将src中有效元素的低半部分bit数据连续存储于dst中，根据数据类型自动选择位宽粒度。 | min(32, VL/2) |
  | pypto_pro.language.StoreDist.PACK_B16 | 压缩模式，根据mask将src中有效元素的低半部分bit数据按16位宽类型连续存储于dst中。 | min(32, VL/2) |
  | pypto_pro.language.StoreDist.PACK_B32 | 压缩模式，根据mask将src中有效元素的低半部分bit数据按32位宽类型连续存储于dst中。 | min(32, VL/2) |
  | pypto_pro.language.StoreDist.PACK_B64 | 压缩模式，根据mask将src中有效元素的低半部分bit数据按64位宽类型连续存储于dst中。 | min(32, VL/2) |
  | pypto_pro.language.StoreDist.PACK4 | 4元素压缩模式（通用），根据mask将src中有效元素的低8bit（四分之一）数据连续存储于dst中。 | min(32, VL/4) |
  | pypto_pro.language.StoreDist.PACK4_B32 | 4元素压缩模式，按32位宽类型粒度将src中有效元素的低8bit（四分之一）数据连续存储于dst中。 | min(32, VL/4) |

  **表2** reg_tensor双搬出模式dist参数说明

  | dist取值 | 含义 | 对齐约束（Byte） |
  |---|---|---|
  | pypto_pro.language.StoreDist.INTLV | 交错存储（通用），将src0、src1中的元素交错存储于dst中，根据数据类型自动选择位宽粒度。 | 32 |
  | pypto_pro.language.StoreDist.INTLV_B8 | 按8位宽类型交错存储，将src0、src1中的元素交错存储于dst中。 | 32 |
  | pypto_pro.language.StoreDist.INTLV_B16 | 按16位宽类型交错存储，将src0、src1中的元素交错存储于dst中。 | 32 |
  | pypto_pro.language.StoreDist.INTLV_B32 | 按32位宽粒度交错存储，将src0、src1中的元素交错存储于dst中。 | 32 |

  **表3** mask_reg模式dist参数说明

  | dist取值 | 含义 | 对齐约束（Byte） |
  |---|---|---|
  | pypto_pro.language.StoreDist.NORM | 正常模式，搬运VL/8数据。 | VL/8 |
  | pypto_pro.language.StoreDist.PACK | 压缩模式，每间隔1bit舍弃数据，将VL/8的数据压缩为VL/16搬出。 | VL/16 |

- 接口调用方式说明：

  **表4** 三类搬运接口调用方式

  | 接口类型 | 触发条件 | 说明 |
  |---|---|---|
  | 普通搬运接口 | offset为整数或[row, col]列表，post_update=False（默认） | 完成一次搬运后，Tile地址不会自动更新，每次迭代需要手动更新offset。 |
  | PostUpdate扩展搬运接口 | post_update=True或传入stride | 完成一次搬运后，Tile地址会自动更新，每次迭代不需要手动更新offset。适用于循环内连续存储。 |
  | AddrReg存储偏移量接口 | offset为vf.create_addr_reg创建的AddrReg | 在每次迭代中，需要先调用vf.create_addr_reg手动设定地址偏移量，再调用搬运指令。 |

## 返回值说明

无。

## 调用示例

### 普通对齐存储

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    vf.store_align(dst_tile, reg, preg)

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
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### interleaved存储示例

使用dist=pypto_pro.language.StoreDist.INTLV_B32将偶数/奇数寄存器交错写入Tile：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    dst_even, dst_odd = vf.load_align(src_tile, 0, dist=pl.LoadDist.DINTLV_B32)
    vf.store_align(dst_tile, dst_even, dst_odd, preg, dist=pl.StoreDist.INTLV_B32)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float32)
    out = torch.empty([1, 128], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### 列表偏移存储示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, [0, 0])
    vf.store_align(dst_tile, reg, preg, [1, 0])

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(t_out, out, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_6():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([2, 64], device=device, dtype=torch.float32)
    out = torch.zeros([2, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = torch.zeros([2, 64], device=device, dtype=torch.float32)
    expected[1, :] = a[0, :]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_6()
    print("PASSED")
```

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile, n_rows):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(n_rows):
        reg = vf.load_align(src_tile, [i, 0])
        vf.store_align(dst_tile, reg, preg, [i, 0])

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out, 2)
        pl.store(out, t_out, [0, 0])

def test_example_7():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([2, 64], device=device, dtype=torch.float32)
    out = torch.empty([2, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_7()
    print("PASSED")
```

### AddrReg偏移存储示例

使用vf.create_addr_reg创建地址偏移寄存器，在循环中同步偏移load和store地址：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_repeat_size = 64
    repeat_times = 2
    for i in pl.range(0, repeat_times, 1):
        a_reg = vf.create_addr_reg(one_repeat_size, dtype=pl.DT_FP32)
        reg = vf.load_align(src_tile, a_reg)
        vf.store_align(dst_tile, reg, preg, a_reg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_3():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float32)
    out = torch.empty([1, 128], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_3()
    print("PASSED")
```

### mask_reg存储示例

当src为mask_reg时，vf.store_align自动分派mask_reg存储路径，无需传入谓词mask参数：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, mask_buf_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    cmp_mask = vf.ge(reg_a, 0.0, preg)
    vf.store_align(mask_buf_tile, cmp_mask, dist=pl.StoreDist.PACK)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    loaded_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    loaded_mask = vf.load_align(mask_buf_tile, dist=pl.LoadDist.US)
    reg_dst = vf.abs(reg_a, loaded_mask)
    vf.store_align(dst_tile, reg_dst, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_mask_grp = pl.make_tile_group(type=tu, addrs=0x100, mutex_ids=[1])
    t_mask = t_mask_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_mask, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_4():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = torch.where(a >= 0, torch.abs(a), torch.tensor(0.0, device=device))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_4()
    print("PASSED")
```

### Post-update连续存储示例

使用post_update=True在循环中连续存储，目标地址自动累进，无需手动计算偏移：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_repeat_size = 64
    repeat_times = 2
    for i in pl.range(0, repeat_times, 1):
        reg = vf.load_align(src_tile, i * one_repeat_size)
        vf.store_align(dst_tile, reg, preg, one_repeat_size, post_update=True)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_5():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float32)
    out = torch.empty([1, 128], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_5()
    print("PASSED")
```

### FP8数据存储示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    preg_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_f8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP8E4M3FN)
    reg_f32 = vf.load_align(src_tile, 0)
    reg_f8 = vf.astype(reg_f32, preg_f32, dtype=pl.DT_FP8E4M3FN,
                       layout=pl.CastLayout.ZERO, saturate=pl.SaturateMode.ON)
    vf.store_align(dst_tile, reg_f8, preg_f8)

@pl.jit()
def example_kernel_fp8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
):
    tf_in = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 256], dtype=pl.DT_FP8E4M3FN, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp8(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 256], device=device, dtype=torch.float8_e4m3fn)
    example_kernel_fp8[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a.to(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(out.to(torch.float32)[:, ::4], expected, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_example_fp8()
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
    vf.store_align(dst_tile, reg_a, preg)

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
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
