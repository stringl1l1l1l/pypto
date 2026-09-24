# pypto_pro.language.matmul_mx_acc

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

在已有累加器值上累加一次带分组量化系数的矩阵乘结果：

$$
C_{M \times N} = \left(\mathrm{scale\_a}_{M \times (K/32)} \otimes A_{M \times K}\right) \times \left(\mathrm{scale\_b}_{(K/32) \times N} \otimes B_{K \times N}\right) + C_{M \times N}
$$

其中，⊗表示广播乘法，A、B沿K维度每连续32个元素分别共享scale_a、scale_b中的一个量化系数。

## 函数原型

```python
pypto_pro.language.matmul_mx_acc(
    dst_tile: Tile,
    acc_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    scale_a: Tile,
    scale_b: Tile,
    *,
    phase: Optional[AccPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L0C Buffer，形状为[M, N]，数据类型为DT_FP32，layout必须为NZ。 |
| acc_tile | 输入 | 源操作数（已有累加结果），Tile类型，存储空间为L0C Buffer，数据类型为DT_FP32，layout必须为NZ，形状必须与dst_tile一致。通常与dst_tile使用同一Tile实现原地累加；其内容应由此前的matmul_mx或matmul_mx_acc初始化。 |
| lhs_tile | 输入 | 源操作数（A，左矩阵），Tile类型，存储空间为L0A Buffer，形状为[M, K]，数据类型支持DT_FP8E4M3FN、DT_FP8E5M2、DT_FP4E2M1和DT_FP4E1M2，layout必须为NZ，K必须与rhs_tile的K维一致且为64的倍数。 |
| rhs_tile | 输入 | 源操作数（B，右矩阵），Tile类型，存储空间为L0B Buffer，形状为[K, N]，数据类型支持DT_FP8E4M3FN、DT_FP8E5M2、DT_FP4E2M1和DT_FP4E1M2。与lhs_tile的数据类型可以不同，但必须同时选自DT_FP8E4M3FN、DT_FP8E5M2，或同时选自DT_FP4E2M1、DT_FP4E1M2。layout必须为ZN，K必须与lhs_tile的K维一致且为64的倍数。 |
| scale_a | 输入 | 源操作数（左量化系数矩阵），Tile类型，位于L0A_MX Buffer，数据类型为DT_FP8E8M0，形状为[M, K/32]，layout默认且仅支持ZZ，fractal默认且仅支持32。每个量化系数对应A矩阵K方向连续32个元素。 |
| scale_b | 输入 | 源操作数（右量化系数矩阵），Tile类型，位于L0B_MX Buffer，数据类型为DT_FP8E8M0，形状为[K/32, N]，layout默认且仅支持NN，fractal默认且仅支持32。每个量化系数对应B矩阵K方向连续32个元素。 |
| phase | 输入 | K维分块累加阶段，[pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md)类型，可选，用于控制矩阵计算与L0C Buffer数据搬出之间的UnitFlag同步。与[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)的配合方式见[Cube计算进阶](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md)。 |

## 约束说明

- 量化系数Tile与A/B矩阵Tile必须满足以下硬件地址关系：

  ```text
  addr(scale_a) = addr(lhs_tile) >> 4
  addr(scale_b) = addr(rhs_tile) >> 4
  ```

  硬件根据A/B矩阵首地址定位scale_a/scale_b。使用多组L0A/L0B Tile时，每组地址均须满足上述关系。

- 矩阵乘累加操作时，acc_tile应调用[matmul_mx](matmul_mx.md)建立初始结果，后续再调用本接口进行累加操作。

## 返回值说明

无。

## 调用示例

将K方向分为两个128元素的块，首块调用matmul_mx，末块调用matmul_mx_acc。A和B的完整K方向均包含8个量化系数分组，因此GM中的scale_a/scale_b Tensor物理形状分别为[128, 4, 2]和[4, 128, 2]。

```python
import os
import pypto_pro.language as pl
import torch

M_SIZE = 128
K_SIZE = 256
N_SIZE = 128
TILE_K = 128
SCALE_K = TILE_K // 32


@pl.jit(auto_mutex=True)
def mxfp8_matmul_acc_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E5M2],
    scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, TILE_K], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, N_SIZE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    scale_a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
        addrs=0x20000, mutex_ids=[2])
    scale_b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, N_SIZE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN),
        addrs=0x21000, mutex_ids=[3])
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, TILE_K], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4])
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, N_SIZE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000, mutex_ids=[5])
    scale_a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
        addrs=0x0000, mutex_ids=[6])
    scale_b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, N_SIZE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
        addrs=0x0000, mutex_ids=[7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, N_SIZE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        al1, bl1 = a_l1.current(), b_l1.current()
        sal1, sbl1 = scale_a_l1.current(), scale_b_l1.current()
        al0, bl0 = a_l0.current(), b_l0.current()
        sal0, sbl0 = scale_a_l0.current(), scale_b_l0.current()
        ac = acc.current()

        for ki in pl.range(0, K_SIZE, TILE_K):
            pl.load(al1, a, [0, ki])
            pl.load(bl1, b, [ki, 0])
            pl.load(sal1, scale_a, [0, ki // 64, 0], order=[0, 1])
            pl.load(sbl1, scale_b, [ki // 64, 0, 0], order=[0, 1])
            pl.move(al0, al1)
            pl.move(bl0, bl1)
            pl.move(sal0, sal1)
            pl.move(sbl0, sbl1)

            if ki == 0:
                pl.matmul_mx(ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_mx_acc(ac, ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Final)

        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    # E4M3编码0x38、E5M2编码0x3C均表示1.0；E8M0编码127表示缩放值1.0。
    a = torch.full([M_SIZE, K_SIZE], 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)
    b = torch.full([K_SIZE, N_SIZE], 0x3C, dtype=torch.uint8).view(torch.float8_e5m2).to(device)
    scale_a = torch.full([M_SIZE, K_SIZE // 64, 2], 127, device=device, dtype=torch.uint8)
    scale_b = torch.full([K_SIZE // 64, N_SIZE, 2], 127, device=device, dtype=torch.uint8)
    out = torch.zeros([M_SIZE, N_SIZE], device=device, dtype=torch.float32)

    mxfp8_matmul_acc_kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()

    ref = torch.full([M_SIZE, N_SIZE], float(K_SIZE), device=device, dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print(f"max diff = {(out - ref).abs().max().item()}")
```
