# vf.shift_left

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

vf.shift_left指令根据preg对源操作数src进行左移操作，将结果写入目的操作数dst。移位量shift既可以是**标量**（所有元素移动相同位数），也可以是**[reg_tensor](../reg_tensor.md)**（每个元素按对应lane的位数移动）。接口会根据shift参数的类型自动选择：

- **标量模式**（整数值或标量变量）：所有元素统一左移。

- **reg_tensor模式**：reg_tensor中逐元素左移。根据源操作数的数据类型，左移操作分为以下两种情况：

- **数据类型为无符号类型：执行逻辑左移。**

  逻辑左移会将二进制数整体向左移动指定的位数，最高位被丢弃，最低位用0填充。例如，二进制数1010101010101010（DT_UINT16类型）逻辑左移1位后，结果为0101010101010100。
- **数据类型为有符号类型：执行算术左移。**

  算术左移与逻辑左移一样，将超出数据类型位宽的最高位丢弃，并在最低位补0；区别只体现在结果按有符号类型解释。例如，二进制数1010101010101010（DT_INT16类型）左移1位后，位模式为0101010101010100；左移3位后，位模式为0101010101010000。

$$
dst_i = src_i \ll shift_i
$$

## 函数原型

```python
shift_left(src, shift, preg, mode: Optional[MergeMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../reg_tensor.md)。源操作数src与目的操作数dst的数据类型保持一致。支持的数据类型参见[约束说明](#约束说明)。 |
| shift | 输入 | 左移位数。标量（整型，所有元素统一移位）或[reg_tensor](../reg_tensor.md)（逐元素移位），支持的数据类型参见[约束说明](#约束说明)。<br>- 对于**reg_tensor模式**下逻辑位移（无符号数据类型），如果位移量大于数据类型位宽，则输出为0。<br>- 对于**reg_tensor模式**下算术位移（有符号数据类型），如果位移量大于数据类型位宽，则输出0。<br>- 两种模式下均不支持设置为负数，负数行为未定义。 |
| preg | 输入 | [mask_reg](../mask_reg.md)。 |
| mode | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.ZEROING（默认），preg未筛选的元素在dst中置0。<br>- pypto_pro.language.MergeMode.MERGING当前不支持。 |

## 约束说明

- 数据类型约束：

  | dst | src | shift |
  | :-- | :-- | :-- |
  | DT_INT8 | DT_INT8 | 标量模式：整型标量；reg_tensor模式：DT_INT8 |
  | DT_UINT8 | DT_UINT8 | 标量模式：整型标量；reg_tensor模式：DT_INT8 |
  | DT_INT16 | DT_INT16 | 标量模式：整型标量；reg_tensor模式：DT_INT16 |
  | DT_UINT16 | DT_UINT16 | 标量模式：整型标量；reg_tensor模式：DT_INT16 |
  | DT_INT32 | DT_INT32 | 标量模式：整型标量；reg_tensor模式：DT_INT32 |
  | DT_UINT32 | DT_UINT32 | 标量模式：整型标量；reg_tensor模式：DT_INT32 |
  | DT_INT64 | DT_INT64 | 标量模式：整型标量；reg_tensor模式：DT_INT64 |
  | DT_UINT64 | DT_UINT64 | 标量模式：整型标量；reg_tensor模式：DT_INT64 |

## 返回值说明

返回dst目的操作数，[reg_tensor](../reg_tensor.md)，支持的数据类型参见[约束说明](#约束说明)。

## 调用示例

### 标量模式

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_scalar(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_out = vf.shift_left(reg_src, 4, preg)
    vf.store_align(dst_tile, reg_out, preg)

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
        example_vf_scalar(in_a, t_out)
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
    torch.testing.assert_close(out, a << 4, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### reg_tensor模式

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_vector(src_tile, shift_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_shift = vf.load_align(shift_tile, 0)
    reg_out = vf.shift_left(reg_src, reg_shift, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_vector(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    shift: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf_u32 = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tf_i32 = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_u32, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_shift_grp = pl.make_tile_group(type=tf_i32, addrs=0x100, mutex_ids=[1])
    in_shift = in_shift_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_u32, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_shift, shift, [0, 0])
        example_vf_vector(in_a, in_shift, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 100, [1, 64], device=device, dtype=torch.int32)
    shift = torch.full([1, 64], 4, device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel_vector[None, core_nums](a, shift, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a << 4, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_2()
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
    reg_out = vf.shift_left(reg_a, 2, preg)
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
    torch.testing.assert_close(out, a << 2, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
