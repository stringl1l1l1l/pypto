# vf.load_align

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

从Tile对齐加载数据到寄存器，支持reg_tensor和mask_reg两种寄存器类型：

### reg_tensor模式

reg_tensor模式支持**连续搬运模式**和**非连续搬运模式**。**连续搬运模式**又分为两种模式，单搬入模式和双搬入模式：

- **reg_tensor单搬入模式**：从Tile读取VL（寄存器长度）数据量，搬入到一个reg_tensor中。
- **reg_tensor双搬入模式**（de-interleave）：从Tile读取2*VL数据量，交错搬运，将偶数索引和奇数索引的元素分别搬入两个reg_tensor（2*VL）中。

连续数据搬入时，可以通过dist关键字参数配置搬运的数据分布模式，能够实现broadcast、上采样、下采样、解压缩等功能。下图展示了连续搬入模式。完整的dist取值及对齐约束请参见[约束说明](#约束说明)。

**图1** vf.load_align连续对齐搬入分布模式图示

![](../../../figures/contiguous_aligned_load.jpg)

**非连续搬运模式**下，通过data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY启用DataBlock搬运模式，实现数据从Tile非连续搬运至reg_tensor。单条指令一次搬运8个DataBlock，block_stride参数表示相邻DataBlock间的间隔。该模式下offset参数改为传入控制有效元素的mask_reg，控制规则如下：

- 某个DataBlock在mask中对应的32bit有任意一位为1时，该DataBlock对应的数据会被搬入dst。
- 某个DataBlock在mask中对应的32bit全为0时，该DataBlock对应的数据不会被读取，dst对应位置置0，即使Tile越界也不会报错。

### mask_reg模式

当目标变量已通过vf.create_mask预声明为mask_reg时，后端自动分派mask_reg加载路径，实现数据从Tile搬运至mask_reg。

mask_reg目标支持三种分布模式，通过dist关键字参数配置，能够实现上采样、下采样等功能：

- pypto_pro.language.LoadDist.NORM（正常模式，搬运数据量为VL/8）
- pypto_pro.language.LoadDist.US（上采样模式，每bit数据重复搬运两次，将VL/16数据扩充为VL/8搬入）
- pypto_pro.language.LoadDist.DS（下采样模式，每间隔1bit舍弃数据，将VL/4数据压缩为VL/8搬入）

各模式示意如下：

**图2** pypto_pro.language.LoadDist.NORM模式

![LoadDist-NORM模式](../../../figures/load_dist_norm_mode.jpg)

**图3** pypto_pro.language.LoadDist.US模式

![LoadDist-US模式](../../../figures/load_dist_us_mode.jpg)

**图4** pypto_pro.language.LoadDist.DS模式

![LoadDist-DS模式](../../../figures/load_dist_ds_mode.jpg)

## 函数原型

```python
load_align(tile, offset=None, dist: Optional[LoadDist] = None, dtype: Optional[DType] = None, post_update: bool = False, block_stride=None, repeat_stride=None, data_copy_mode: Optional[DataCopyMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tile | 输入 | 源操作数，Tile地址。地址需要32字节对齐。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| offset | 输入 | 可选，地址偏移参数，根据传入类型自动分派搬运接口。在**连续搬运模式**和**mask_reg模式**下单位为元素个数，在**非连续搬运模式**时为mask。<br>- **连续搬运模式**：<br>&nbsp;&nbsp;- **整数或[row, col]列表**：整数偏移在代码生成时转换为指针算术Tile + offset。<br>&nbsp;&nbsp;&nbsp;&nbsp;[row, col]列表的线性偏移为row * shape[1] + col，row单位为Tile的列数（即set_validshape[m, n]的n），col单位为元素个数，两者均支持表达式。<br>&nbsp;&nbsp;- **AddrReg**（由vf.create_addr_reg创建）：实际搬运Tile地址为Tile + AddrReg中存储的偏移量。<br>&nbsp;&nbsp;&nbsp;&nbsp;每次迭代需先调用vf.create_addr_reg设定偏移量再调用搬运指令。<br>- **非连续搬运模式**（DataBlock加载模式，data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY时）：该模式下offset位置改为传入控制有效元素的mask_reg，<br>&nbsp;&nbsp;某个DataBlock在mask中对应的32bit有任意一位为1时搬入，全为0时不读取且dst对应位置置0。<br>&nbsp;&nbsp;- 当post_update=True时，搬运后源地址自动累进repeat_stride步长，每次迭代无需手动更新地址。<br>&nbsp;&nbsp;- 当post_update=False时，搬运后地址不更新。<br>- **mask_reg模式**（需先用vf.create_mask预声明）：<br>&nbsp;&nbsp;- **整数**：仅在post_update=True时生效；post_update=False时不支持整数offset。<br>&nbsp;&nbsp;- **AddrReg**（由vf.create_addr_reg创建）：实际搬运Tile地址为srcAddr + AddrReg中存储的偏移量。 |
| dist | 输入 | 可选，数据分布模式，对应[LoadDist](../types/LoadDist.md)类型，具体模式根据是**reg_tensor单搬入模式**、**reg_tensor双搬入模式**还是**mask_reg模式**请分别参见[约束说明](#约束说明)中各表。 |
| dtype | 输入 | 可选，指定目标[reg_tensor](../reg_tensor.md)或者[mask_reg](../mask_reg.md)的数据类型。当源Tile的数据类型与期望的寄存器数据类型不一致时需要指定（例如源Tile为DT_FP32但需要按DT_UINT32位重解释加载到寄存器）。默认从源Tile的数据类型推断。 |
| post_update | 输入 | 可选，True时搬运后源地址自动累进，默认False。适用于循环内连续加载，避免手动更新offset。 |
| block_stride | 输入 | 可选，仅在data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY模式下有效，其他模式下传入会被忽略。表示相邻DataBlock间的间隔，单位：DataBlock（32字节）。当block_stride=0时，表示重复搬入第一个DataBlock。 |
| repeat_stride | 输入 | 可选，仅在data_copy_mode=pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY模式且post_update=True时有效。表示重复搬运时的地址更新步长，单位：DataBlock（32字节），需要32字节对齐。post_update=True时，搬运后源地址自动更新为srcAddr += repeat_stride * 32B。 |
| data_copy_mode | 输入 | 可选，数据拷贝模式，对应[DataCopyMode](../types/DataCopyMode.md)类型。仅在**reg_tensor模式**下有效，mask_reg目标不支持此参数。取值：pypto_pro.language.DataCopyMode.NORM（默认，普通连续搬运）或pypto_pro.language.DataCopyMode.DATA_BLOCK_COPY（非连续以DataBlock（32B）为单位进行搬运）。 |

## 约束说明

- 各模式下dist参数说明：

  部分模式提供通用形式和显式粒度形式。通用形式（如BRC、US）不带粒度后缀，后端会根据目标寄存器的数据类型自动选择对应的B8/B16/B32变体；显式粒度形式（如BRC_B8、US_B16）直接指定粒度。两种形式均可使用。完整的取值说明请参见[LoadDist枚举类型](../types/LoadDist.md)。

  **表1** reg_tensor单搬入模式dist参数说明

  | dist取值 | 含义 | 搬运对齐约束（Byte） |
  |---|---|---|
  | pypto_pro.language.LoadDist.NORM | 正常模式，搬运VL数据量。 | 32 |
  | pypto_pro.language.LoadDist.BRC | 广播模式（通用），根据数据类型自动选择B8/B16/B32粒度。 | 1/2/4 |
  | pypto_pro.language.LoadDist.BRC_B8 | 搬运一个8位宽类型的数据，并Broadcast到所有元素位置。 | 1 |
  | pypto_pro.language.LoadDist.BRC_B16 | 搬运一个16位宽类型的数据，并Broadcast到所有元素位置。 | 2 |
  | pypto_pro.language.LoadDist.BRC_B32 | 搬运一个32位宽类型的数据，并Broadcast到所有元素位置。 | 4 |
  | pypto_pro.language.LoadDist.US | 上采样模式（通用），根据数据类型自动选择B8/B16粒度。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.US_B8 | 数据2倍上采样，加载数据量为VL/2，每个输入元素重复两次，数据类型为8位宽类型。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.US_B16 | 数据2倍上采样，加载数据量为VL/2，每个输入元素重复两次，数据类型为16位宽类型。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.DS | 下采样模式（通用），根据数据类型自动选择B8/B16粒度。 | 32 |
  | pypto_pro.language.LoadDist.DS_B8 | 数据2倍下采样，加载数据量为2*VL，数据每隔一个保留，数据类型为8位宽类型。 | 32 |
  | pypto_pro.language.LoadDist.DS_B16 | 数据2倍下采样，加载数据量为2*VL，数据每隔一个保留，数据类型为16位宽类型。 | 32 |
  | pypto_pro.language.LoadDist.UNPK | 解压缩模式（通用），根据数据类型自动选择B8/B16/B32粒度。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.UNPK_B8 | 解压缩模式，按无符号整型u8加载VL/2数据量，每个元素后会补1个值为0元素，即unpack到VL。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.UNPK_B16 | 解压缩模式，按无符号整型u16加载VL/2数据量，每个元素后会补1个值为0元素，即unpack到VL。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.UNPK_B32 | 解压缩模式，按无符号整型u32加载VL/2数据量，每个元素后会补1个值为0元素，即unpack到VL。 | min(32, VL/2) |
  | pypto_pro.language.LoadDist.UNPK4 | 4元素解压缩模式，固定按u8加载VL/4数据量，unpack到VL，每个元素后会补3个值为0的元素。 | min(32, VL/4) |
  | pypto_pro.language.LoadDist.BLK | 读取一个DataBlock（32B），并广播到VL。 | 32 |
  | pypto_pro.language.LoadDist.E2B | 元素扩展到DataBlock模式（通用），根据数据类型自动选择B16/B32粒度。 | VL/16或VL/8 |
  | pypto_pro.language.LoadDist.E2B_B16 | 加载(VL/DataBlock)个B16的数据，并将每个元素（16bit）广播到一个DataBlock（32B）中。 | VL/16 |
  | pypto_pro.language.LoadDist.E2B_B32 | 加载(VL/DataBlock)个B32的数据，并将每个元素（32bit）广播到一个DataBlock（32B）中。 | VL/8 |

  **表2** reg_tensor双搬入模式dist参数说明

  | dist取值 | 含义 | 搬运对齐约束（Byte） |
  |---|---|---|
  | pypto_pro.language.LoadDist.DINTLV_B8 | 双搬入模式，基于元素的交错搬运，将偶数索引的元素存入dst0，奇数索引的元素存入dst1，数据类型为8位宽类型。 | 32 |
  | pypto_pro.language.LoadDist.DINTLV_B16 | 双搬入模式，基于元素的交错搬运，将偶数索引的元素存入dst0，奇数索引的元素存入dst1，数据类型为16位宽类型。 | 32 |
  | pypto_pro.language.LoadDist.DINTLV_B32 | 双搬入模式，基于元素的交错搬运，将偶数索引的元素存入dst0，奇数索引的元素存入dst1，数据类型为32位宽类型。 | 32 |

  **表3** mask_reg模式dist参数说明

  | dist取值 | 含义 | 搬运对齐约束（Byte） |
  |---|---|---|
  | pypto_pro.language.LoadDist.NORM | 正常模式，搬运数据量为VL/8。 | VL/8 |
  | pypto_pro.language.LoadDist.US | 上采样模式，每bit数据重复搬运两次，将VL/16数据扩充为VL/8搬入。 | VL/16 |
  | pypto_pro.language.LoadDist.DS | 下采样模式，每间隔1bit舍弃数据，将VL/4数据压缩为VL/8搬入。 | min(32, VL/4) |

## 返回值说明

返回dst目的操作数，[reg_tensor](../reg_tensor.md)或者[mask_reg](../mask_reg.md)类型。

- 当目标为reg_tensor时，为**reg_tensor单搬入模式**，支持的数据类型和Tile中的说明一致。

- dst_even/dst_odd**reg_tensor双搬入模式**的偶数/奇数目的操作数，[reg_tensor](../reg_tensor.md)，支持的数据类型和Tile中的说明一致。

- 当目标已通过vf.create_mask预声明为mask_reg时，自动分派mask_reg加载路径，将Tile中的数据搬入mask_reg。

## 调用示例

### 普通对齐加载

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src0 = vf.load_align(src_tile, 0)
    vf.store_align(dst_tile, src0, preg)
    reg = vf.load_align(src_tile, 0, post_update=True)
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

### de-interleave加载示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    dst_even, dst_odd = vf.load_align(src_tile, 0, dist=pl.LoadDist.DINTLV_B32)
    vf.store_align(dst_tile, dst_even, preg)

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
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a[:, ::2]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### 列表偏移加载示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, [1, 0])
    vf.store_align(dst_tile, reg, preg)

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
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_5():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([2, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[1:2, :], rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_5()
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
        vf.store_align(dst_tile + i * 64, reg, preg)

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

def test_example_6():
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
    test_example_6()
    print("PASSED")
```

### AddrReg偏移加载示例

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

### mask_reg加载示例

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

### FP8 数据加载示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_f8 = vf.load_align(src_tile, 0, dtype=pl.DT_FP8E4M3FN)
    reg_f32 = vf.astype(reg_f8, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel_fp8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_FP8E4M3FN, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
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
    a = torch.randn([1, 256], device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp8[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a.to(torch.float32)
    torch.testing.assert_close(out, expected[:, ::4], rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_example_fp8()
    print("PASSED")
```

### HF8数据加载与存储

以HF8类型加载数据并直接存回，验证HF8类型的搬运路径。HF8为8位存储类型，需使用b8宽度的mask。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8_store(src_tile, dst_tile):
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_HF8)
    reg_hf8 = vf.load_align(src_tile, 0, dtype=pl.DT_HF8)
    vf.store_align(dst_tile, reg_hf8, preg_b8)

@pl.jit()
def example_kernel_hf8_store(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_hf8_store(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_hf8_store():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 256, [1, 256], device=device, dtype=torch.uint8)
    out = torch.empty([1, 256], device=device, dtype=torch.uint8)
    example_kernel_hf8_store[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_hf8_store()
    print("PASSED")
```

### HF8转FP32（layout=TWO/THREE）

HF8为8位类型，FP32为32位类型，二者位宽比为1:4。FP32→HF8为4x缩窄转换，layout参数控制转换结果放在HF8寄存器的哪个子区（CastLayout.ZERO/ONE/TWO/THREE分别对应第0/1/2/3个byte位置）。HF8→FP32为4x扩展转换，layout参数控制从HF8寄存器的哪个子区读取数据扩展为FP32。

以下示例先通过FP32→HF8转换（layout=ZERO）生成HF8数据，再分别使用layout=ZERO/ONE/TWO/THREE将HF8数据转换回FP32并存储，展示四种CastLayout的使用方式。其中layout=ZERO读取有效数据，其余layout读取未写入的子区，结果为0。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8_to_fp32(src_tile, dst_tile):
    preg_b32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_f32 = vf.load_align(src_tile, 0)
    reg_hf8 = vf.astype(reg_f32, preg_b32, dtype=pl.DT_HF8, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_ROUND, saturate=pl.SaturateMode.ON)
    reg_f32_zero = vf.astype(reg_hf8, preg_b32, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    vf.store_align(dst_tile, reg_f32_zero, preg_b32, 0)
    reg_f32_one = vf.astype(reg_hf8, preg_b32, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    vf.store_align(dst_tile, reg_f32_one, preg_b32, 64)
    reg_f32_two = vf.astype(reg_hf8, preg_b32, dtype=pl.DT_FP32, layout=pl.CastLayout.TWO)
    vf.store_align(dst_tile, reg_f32_two, preg_b32, 128)
    reg_f32_three = vf.astype(reg_hf8, preg_b32, dtype=pl.DT_FP32, layout=pl.CastLayout.THREE)
    vf.store_align(dst_tile, reg_f32_three, preg_b32, 192)

@pl.jit()
def example_kernel_hf8_to_fp32(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf_in = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_hf8_to_fp32(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_hf8_to_fp32():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 256], device=device, dtype=torch.float32)
    example_kernel_hf8_to_fp32[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:, 0:64], a, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(out[:, 64:256], torch.zeros([1, 192], device=device, dtype=torch.float32), rtol=0, atol=0)

if __name__ == "__main__":
    test_example_hf8_to_fp32()
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
    reg_out = vf.load_align(src_tile, 0)
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
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
