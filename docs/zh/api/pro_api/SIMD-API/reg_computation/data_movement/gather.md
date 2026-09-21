# vf.gather

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

该指令会根据索引值index将源操作数收集到目的操作数dst中。后端根据src参数类型自动分发为两种形式：

**Tile→reg形式**（src为Tile）：从Tile中按索引收集数据到reg_tensor。通过data_copy_mode选择收集粒度：

- pypto_pro.language.DataCopyMode.NORM（默认）：NORM模式，按元素收集，index单位为元素。收集过程如下图所示：

  **图1** gather NORM模式功能说明

  ![](../../../figures/gather_function.jpg)

- pypto_pro.language.DataCopyMode.DATA_BLOCK_LOAD：DATA_BLOCK_LOAD模式，按DataBlock（32B）收集，index单位为字节且需32B对齐。收集过程如下图所示：

  **图2** gather DATA_BLOCK_LOAD模式功能说明

  ![](../../../figures/block_mode_gather.jpg)

**reg→reg形式**（src为reg_tensor）：reg_tensor到reg_tensor按元素收集，无需mask。

> **两种形式的区别**：Tile→reg形式从Tile中读取数据，8位宽的数据类型（DT_INT8、DT_UINT8）源数据会被零扩展到16位宽；reg→reg形式从reg_tensor读取数据，保持源数据类型不变。

## 函数原型

```python
gather(src, index, preg, data_copy_mode: Optional[DataCopyMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，可为Tile（Tile→reg形式，基地址需32字节对齐）或[reg_tensor](../reg_tensor.md)（reg→reg形式），支持的数据类型请参见[约束说明](#约束说明)。<br>- NORM模式下，当src为16位宽数据类型且index为DT_UINT32时，每个gather到的16位元素占32位空间（低16位为数据，高16位补零），寄存器中有效元素数量为索引数量（VL/4），而非index为DT_UINT16索引场景的VL/2。存储时需使用pypto_pro.language.StoreDist.NORM_B16按16位粒度写入，输出中偶数位置为有效数据，奇数位置为零。此功能适用于索引数据天然为32位宽的场景。<br>- DATA_BLOCK_LOAD模式和reg→reg形式下，源操作数和目的操作数数据类型必须相同。 |
| index | 输入 | 索引值，[reg_tensor](../reg_tensor.md)，支持的数据类型请参见[约束说明](#约束说明)。<br>- Tile→reg NORM模式下为dst中每个元素相对于src的位置，单位：元素。8位宽的数据类型（DT_INT8、DT_UINT8）源数据会被零扩展到16位宽。16位宽源数据类型（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）的索引支持DT_UINT16和DT_UINT32。<br>- Tile→reg DATA_BLOCK_LOAD模式下为每个DataBlock相对于src的位置，且必须32B对齐，即一个索引值对应1个DataBlock。<br>- reg→reg形式下为src中每个元素的位置，单位：元素，数据类型位宽需与src保持一致。<br>index索引值对应的数据必须在Tile有效地址范围内（Tile→reg形式）。如果索引值超出当前reg_tensor中能存储的最大数据元素个数，索引值更新为i % (VL / sizeof(T))，其中VL为256字节（reg→reg形式）。index中的值可以重复。 |
| preg | 输入 | 可选，[mask_reg](../mask_reg.md)。mask功能**仅Tile→reg形式支持**，reg→reg形式不支持此参数。 |
| data_copy_mode | 输入 | 可选关键字参数，收集粒度。pypto_pro.language.DataCopyMode.NORM（默认，按元素）或pypto_pro.language.DataCopyMode.DATA_BLOCK_LOAD（按32B DataBlock）。**仅Tile→reg形式支持**，reg→reg形式不支持此参数。 |

## 约束说明

- 数据类型约束：

  - **Tile→reg形式 NORM模式（按元素）**

    | dst | src | index |
    |---|---|---|
    | DT_INT16 | DT_INT8 | DT_UINT16 |
    | DT_INT16 | DT_INT16 | DT_UINT16 |
    | DT_UINT16 | DT_UINT8 | DT_UINT16 |
    | DT_UINT16 | DT_UINT16 | DT_UINT16 |
    | DT_FP16 | DT_FP16 | DT_UINT16 |
    | DT_BF16 | DT_BF16 | DT_UINT16 |
    | DT_INT32 | DT_INT32 | DT_UINT32 |
    | DT_UINT32 | DT_UINT32 | DT_UINT32 |
    | DT_FP32 | DT_FP32 | DT_UINT32 |
    | DT_INT64 | DT_INT64 | DT_UINT32 |
    | DT_INT64 | DT_INT64 | DT_UINT64 |
    | DT_UINT64 | DT_UINT64 | DT_UINT32 |
    | DT_UINT64 | DT_UINT64 | DT_UINT64 |
    | DT_INT16 | DT_INT16 | DT_UINT32 |
    | DT_UINT16 | DT_UINT16 | DT_UINT32 |
    | DT_FP16 | DT_FP16 | DT_UINT32 |
    | DT_BF16 | DT_BF16 | DT_UINT32 |

  - **Tile→reg形式 DATA_BLOCK_LOAD模式（按32B DataBlock）**

    支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。
    索引值支持的数据类型为：DT_UINT32。

  - **reg→reg形式**

    支持的数据类型为：8位宽（DT_INT8、DT_UINT8）、16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）、32位宽（DT_INT32、DT_UINT32、DT_FP32）。
    索引值支持的数据类型为：DT_UINT8、DT_UINT16、DT_UINT32。

## 返回值说明

返回dst目的操作数，[reg_tensor](../reg_tensor.md)，支持的数据类型请参见[约束说明](#约束说明)。NORM模式下，当dst为16位宽数据类型（DT_INT16、DT_UINT16、DT_FP16、DT_BF16），src为8位宽数据类型（DT_INT8、DT_UINT8）时，目的操作数的低8位与源操作数相同，高8位自动补0。

## 调用示例

### Tile→reg形式（DT_FP32数据 NORM模式）

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    index_reg = vf.load_align(index_tile, 0)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    idx = torch.arange(64, device=device, dtype=torch.int32).reshape([1, 64])
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### Tile→reg形式（DT_FP32数据 DATA_BLOCK_LOAD模式）

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_datablock(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_idx = vf.arange(0, dtype=pl.DT_UINT32)
    preg_u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_idx_b = vf.shift_left(reg_idx, 5, preg_u)
    dst_reg = vf.gather(src_tile, reg_idx_b, preg,
                        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)
    vf.store_align(dst_tile, dst_reg, preg)

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
        example_vf_datablock(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
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
    test_example_2()
    print("PASSED")
```

### Tile→reg形式（DT_INT8数据 NORM模式）

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int8(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT16)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT16)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg, dist=pl.StoreDist.NORM_B16)

@pl.jit()
def example_kernel_int8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT16],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 128], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_int8(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-128, 127, [1, 256], device=device, dtype=torch.int8)
    idx = torch.arange(128, device=device, dtype=torch.int32).reshape([1, 128]).to(torch.uint16)
    out = torch.empty([1, 128], device=device, dtype=torch.int16)
    example_kernel_int8[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    expected = a[:, :128].to(torch.uint8).to(torch.int16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int8()
    print("PASSED")
```

### Tile→reg形式（DT_FP16数据 NORM模式）

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp16(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT16)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_fp16(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp16():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    idx = torch.arange(128, device=device, dtype=torch.int32).reshape([1, 128]).to(torch.uint16)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_fp16[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_fp16()
    print("PASSED")
```

### Tile→reg形式（DT_INT8数据 DATA_BLOCK_LOAD模式）

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int8_datablock(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_idx = vf.arange(0, dtype=pl.DT_UINT32)
    reg_idx_b = vf.shift_left(reg_idx, 5, preg_u32)
    dst_reg = vf.gather(src_tile, reg_idx_b, preg,
                        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel_int8_db(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int8_datablock(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int8_db():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-128, 127, [1, 256], device=device, dtype=torch.int8)
    out = torch.empty([1, 256], device=device, dtype=torch.int8)
    example_kernel_int8_db[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int8_db()
    print("PASSED")
```

### reg→reg形式

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_gather_reg(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    src_reg = vf.load_align(src_tile, 0)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT8)
    dst_reg = vf.gather(src_reg, index_reg)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel_gather_reg(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_gather_reg(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_gather_reg():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-128, 127, [1, 256], device=device, dtype=torch.int8)
    idx = torch.arange(128, device=device, dtype=torch.int32).to(torch.uint8).reshape([1, 128])
    out = torch.empty([1, 128], device=device, dtype=torch.int8)
    example_kernel_gather_reg[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[:, :128], rtol=0, atol=0)

if __name__ == "__main__":
    test_example_gather_reg()
    print("PASSED")
```

### Tile→reg形式（DT_FP16数据 NORM模式 DT_UINT32索引）

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp16_bc(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT32)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg, dist=pl.StoreDist.NORM_B16)

@pl.jit()
def example_kernel_fp16_bc(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_fp16_bc(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp16_bc():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    idx = torch.arange(64, device=device, dtype=torch.int32).reshape([1, 64])
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_fp16_bc[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    expected = torch.zeros([1, 128], device=device, dtype=torch.float16)
    expected[:, ::2] = a[:, :64]
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_fp16_bc()
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
    reg_idx = vf.arange(0, dtype=pl.DT_UINT32)
    reg_out = vf.gather(src_tile, reg_idx, preg)
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
