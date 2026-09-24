# vf.bit_cast

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

对向量寄存器进行按位类型强转，不进行任何数值转换。

该接口支持两种使用方式：

- **赋值形式**：dst = vf.bit_cast(src, dtype=xxx) — 将src按位重解释为dtype类型，赋值给新声明的dst寄存器。
- **嵌套参数形式**：vf.xor(vf.bit_cast(reg_a, dtype=pypto_pro.language.DT_UINT32), vf.bit_cast(reg_b, dtype=pypto_pro.language.DT_UINT32), preg) — 作为其他vf.xxx调用的参数，满足指令对操作数类型的要求。

与vf.astype的区别：

- vf.astype执行真正的数值类型转换（如DT_FP32→DT_FP16），元素的值会发生变化。
- vf.bit_cast仅改变类型标签，比特模式不变。

## 函数原型

```python
bit_cast(src, dtype: DType) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)。 |
| dtype | 输入 | 目标数据类型。指定src被重解释为的数据类型。 |

## 约束说明

- 源操作数和目标操作数的位宽必须相同（即src和dtype的GetBit()返回值一致），否则行为未定义。

## 返回值说明

返回dst目标操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)。数据类型由dtype参数决定。

## 调用示例

### 赋值形式：FP32转为UINT32

使用赋值形式将FP32寄存器按位重解释为UINT32，再执行按位异或运算。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_bit_cast_assign(src_tile_a, src_tile_b, dst_tile):
    preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_a = vf.load_align(src_tile_a, 0)
    reg_b = vf.load_align(src_tile_b, 0)
    reg_a_u32 = vf.bit_cast(reg_a, dtype=pl.DT_UINT32)
    reg_b_u32 = vf.bit_cast(reg_b, dtype=pl.DT_UINT32)
    reg_c = vf.xor(reg_a_u32, reg_b_u32, preg_u32)
    vf.store_align(dst_tile, reg_c, preg_u32)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf_bit_cast_assign(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    assert torch.equal(out.view(torch.int32), a.view(torch.int32) ^ b.view(torch.int32))

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### 嵌套参数形式：HF8转为UINT8进行按位运算

将DT_HF8寄存器按位重解释为DT_UINT8，然后执行vf.or_按位或运算。HF8为8位存储类型，加载后RegTensor包含256个元素；vf.bit_cast将其按位重解释为DT_UINT8（同样256个元素），vf.or_对UINT8寄存器执行按位或。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8_to_uint8(src_tile_a, src_tile_b, dst_tile):
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    reg_a = vf.load_align(src_tile_a, 0, dtype=pl.DT_HF8)
    reg_b = vf.load_align(src_tile_b, 0, dtype=pl.DT_HF8)
    reg_c = vf.or_(vf.bit_cast(reg_a, dtype=pl.DT_UINT8),
                   vf.bit_cast(reg_b, dtype=pl.DT_UINT8), preg_b8)
    vf.store_align(dst_tile, reg_c, preg_b8)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf_in, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_in, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf_hf8_to_uint8(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_hf8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 256], device=device, dtype=torch.float32)
    a = torch_npu.npu_dtype_cast(a, torch_npu.hifloat8)
    b = torch.randn([1, 256], device=device, dtype=torch.float32)
    b = torch_npu.npu_dtype_cast(b, torch_npu.hifloat8)
    out = torch.empty([1, 256], device=device, dtype=torch.uint8)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    expected = a.view(torch.uint8) | b.view(torch.uint8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_hf8()
    print("PASSED")
```
