# pypto_pro.language.reinterpret

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

把一个已分配的Tile按新的dtype/shape/layout重新声明，返回指向同一地址的Tile新别名，原Tile不受影响。

## 函数原型

```python
pypto_pro.language.reinterpret(
    tile: Union[Tile, TileGroup],
    *,
    dtype: Optional[DType] = None,
    shape: Optional[List[int]] = None,
    layout: Optional[TensorLayout] = None,
) -> Union[Tile, TileGroup]
```

## 参数说明

| 参数   | 输入/输出 | 说明                                                                                                                                                    |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tile   | 输入      | 待重声明的对象，Tile或TileGroup类型。源Tile必须已绑定编译期可确定的Buffer地址；新别名与源对象复用同一地址和大小，不执行数据搬运或类型转换。 |
| dtype  | 输入      | 目标数据类型，[pypto_pro.language.DataType](../basic_data_structures/DataType.md)类型，可选，省略时继承原dtype。指定dtype时必须同时指定shape，且Tile基地址必须按新dtype的元素字节数对齐。 |
| shape  | 输入      | 目标形状，List[int]类型，可选，必须是非空的编译期整数列表，省略时继承原shape。新shape与dtype决定的存储占用不得超过原Tile的Buffer大小；运行时有效形状应使用pypto_pro.language.set_validshape设置。 |
| layout | 输入      | 目标数据排布，[pypto_pro.language.TensorLayout](../basic_data_structures/TensorLayout.md)类型，可选，省略时继承原layout。调用方必须确保Buffer中的物理数据确实符合新layout。 |

## 约束说明

- dtype、shape和layout三个可选参数中至少必须指定一项。
- 原始Tile的pypto_pro.language.set_validshape不继承。
- TileGroup重声明后，与原TileGroup使用同一buffer管理，任意一方的next()行为都会影响buffer的轮转。需要独立轮转buffer时建议使用group[i] [pypto_pro.language.make_tile_group](make_tile_group.md)。

## 返回值说明

在原地址、原大小上重声明的新Tile或TileGroup的别名。

## 调用示例

以下示例将reinterpret的三种典型用法合并到一个可完整运行的脚本中：

1. **dtype重声明**：同一块数据按新类型读写，不做数值转换。
2. **dtype位宽变化**：改变数据类型后元素数随之变化，shape必须随dtype一起重新声明。
3. **layout变换**：NZ重声明为ZN后，直接作为matmul右矩阵实现转置，省一次显式转置搬运。

```python
import os
import pypto_pro.language as pl
import torch

TILE = 64


# ============ 用法1：dtype 重声明（同位宽，FP32 -> INT32） ============
# 把 FP32 Tile 重声明为 INT32，地址和大小都不变，数据逐位保持一致。
@pl.jit(auto_mutex=True)
def dtype_reinterpret_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_in = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        t = tile_in.current()
        pl.load(t, x, [0, 0])
        t2 = pl.reinterpret(t, shape=[TILE, TILE], dtype=pl.DT_INT32)
        pl.store(out, t2, [0, 0])


# ============ 用法2：dtype 位宽变化（FP32 -> FP16，元素数翻倍） ============
# FP32 是 32 位，FP16 是 16 位，重声明后元素数翻倍，shape 必须随 dtype 一起重新声明。
@pl.jit(auto_mutex=True)
def width_change_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[128, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_in = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        t = tile_in.current()
        pl.load(t, x, [0, 0])
        widened = pl.reinterpret(t, shape=[TILE * 2, TILE], dtype=pl.DT_FP16)
        pl.store(out, widened, [0, 0])


# ============ 用法3：layout 变换（NZ -> ZN，matmul 右矩阵转置） ============
# 同一块数据声明为 ZN后，可直接作为 matmul 右矩阵，效果等同于使用其转置。
@pl.jit(auto_mutex=True)
def layout_reinterpret_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m0, k0 = TILE, TILE
    mat_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    t1 = pl.make_tile_group(type=mat_type, addrs=0x0000, mutex_ids=[0])
    t2 = pl.make_tile_group(type=mat_type, addrs=0x10000, mutex_ids=[1])

    left_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    right_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    acc_type = pl.TileType(
        shape=[m0, m0], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
    )
    l0a = pl.make_tile_group(type=left_type, addrs=0x0000, mutex_ids=[2])
    l0b = pl.make_tile_group(type=right_type, addrs=0x0000, mutex_ids=[3])
    acc = pl.make_tile_group(type=acc_type, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        pl.load_tile(t1.current(), a, [0, 0])
        pl.load_tile(t2.current(), a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        t2r = pl.reinterpret(t2.current(), shape=[k0, m0], layout=pl.TensorLayout.ZN)
        pl.move(l0a.current(), t1.current())
        pl.move(l0b.current(), t2r)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc.current(), l0a.current(), l0b.current())
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(out, acc.current(), [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    # 用法1：同位宽 dtype 重声明
    x1 = torch.randn([64, 64], device=device, dtype=torch.float32)
    o1 = torch.zeros([64, 64], device=device, dtype=torch.int32)
    dtype_reinterpret_kernel[None, 1](x1, o1)
    torch.npu.synchronize()
    torch.testing.assert_close(o1.cpu(), x1.cpu().view(torch.int32), rtol=0, atol=0)
    print("用法1 dtype 重声明 PASSED")

    # 用法2：位宽变化
    x2 = torch.randn([64, 64], device=device, dtype=torch.float32)
    o2 = torch.zeros([128, 64], device=device, dtype=torch.float16)
    width_change_kernel[None, 1](x2, o2)
    torch.npu.synchronize()
    torch.testing.assert_close(o2.cpu().view(torch.int32).flatten(), x2.cpu().view(torch.int32).flatten(), rtol=0, atol=0)
    print("用法2 dtype 位宽变化 PASSED")

    # 用法3：layout 变换
    a = torch.randint(-8, 9, [TILE, TILE], device=device, dtype=torch.float16)
    o3 = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    layout_reinterpret_kernel[None, 1](a, o3)
    torch.npu.synchronize()
    got = o3.cpu().float()
    golden = a.cpu().float() @ a.cpu().float().T
    torch.testing.assert_close(got, golden, rtol=1e-2, atol=1e-1)
    print("用法3 layout 变换 PASSED")
```
