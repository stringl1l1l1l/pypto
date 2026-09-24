# pypto_pro.language.matmul_acc

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

在已有累加结果的基础上继续累加一次矩阵乘积：

```text
dst_tile = acc_tile + lhs_tile × rhs_tile
```

## 函数原型

```python
pypto_pro.language.matmul_acc(
    dst_tile: Tile,
    acc_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    *,
    phase: Optional[AccPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L0C Buffer，形状为[M, N]，且形状和数据类型必须与acc_tile一致。数据类型为DT_FP32或DT_INT32，具体类型由乘法输入的数据类型组合决定。layout必须为NZ；未显式指定fractal时，DT_FP32和DT_INT32使用1024。通常与acc_tile指定为同一个Tile，以执行原地累加。M、N的有效取值范围均为[1, 4095]。 |
| acc_tile | 输入 | 源操作数（已有累加结果），Tile类型，存储空间为L0C Buffer。形状和数据类型必须与dst_tile一致，且内容必须由此前的matmul或matmul_acc操作初始化。K维首块应使用matmul初始化，不能直接对未初始化的L0C Buffer执行matmul_acc。 |
| lhs_tile | 输入 | 源操作数（左矩阵），Tile类型，存储空间为L0A Buffer，形状为[M, K]，layout必须为NZ。M、K的有效取值范围均为[1, 4095]。支持的数据类型组合为：DT_FP16 × DT_FP16 → DT_FP32、DT_BF16 × DT_BF16 → DT_FP32、DT_FP32 × DT_FP32 → DT_FP32、DT_INT8 × DT_INT8 → DT_INT32、DT_HF8 × DT_HF8 → DT_FP32；DT_FP8E4M3FN和DT_FP8E5M2可任意两两组合，输出为DT_FP32。 |
| rhs_tile | 输入 | 源操作数（右矩阵），Tile类型，存储空间为L0B Buffer，形状为[K, N]，layout必须为ZN，N的有效取值范围为[1, 4095]。K维必须与lhs_tile一致。数据类型必须与lhs_tile共同满足上述支持组合。 |
| phase | 输入 | K维分块累加阶段，[pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md)类型，可选。使用时必须与store、store_tile或move的[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)正确配对，详见[Cube计算进阶](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md)。 |

## 约束说明

- 启用phase后，非末块使用pypto_pro.language.AccPhase.Partial，末块使用pypto_pro.language.AccPhase.Final，并与store、store_tile或move的[pypto_pro.language.STPhase](../basic_data_structures/STPhase.md)配合使用。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl

TILE = 128
K_SIZE_ACC = 256     # 分 2 个 TILE 块累加


@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    # L1 Buffer、L0A Buffer和L0B Buffer采用双缓冲（next()轮转）
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    # L0C Buffer：K维累加要求fractal=1024
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE):     # K 维分块（累加）
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)          # 首块写入累加器
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)    # 末块累加（K=2 块）
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
```
