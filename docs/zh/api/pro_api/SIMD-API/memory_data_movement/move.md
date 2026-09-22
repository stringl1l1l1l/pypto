# pypto_pro.language.move

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

在L1 Buffer、L0A Buffer/L0B Buffer、L0C Buffer、UB等各级内存之间提供数据搬运功能，并可在搬运过程中实现随路格式转换和量化激活等操作。

## 函数原型

```python
pypto_pro.language.move(
    dst_tile: Tile,
    src_tile: Tile,
    offset: Optional[Offset] = None,
    *,
    acc_to_vec_mode: Optional[AccToVecMode] = None,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    phase: Optional[STPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，支持的数据类型与分形详见[约束说明](#约束说明)。 |
| src_tile | 输入 | 源操作数，Tile类型，支持的数据类型与分形详见[约束说明](#约束说明)。 |
| offset | 输入 | 可选，表示小Tile在大Tile中的相对位置，格式为[offset_m, offset_n]，单位为元素个数。<br>- 当源操作数的shape >= 目的操作数的shape时，表示从源操作数的第offset_m行offset_n列开始读，数据的搬运量取自目的操作数的valid_shape。<br>- 当源操作数的shape < 目的操作数的shape时，表示从目的操作数的第offset_m行offset_n列开始写，数据的搬运量取自源操作数的valid_shape。 |
| acc_to_vec_mode | 输入 | 可选，L0C Buffer→UB搬运时是否开启双目标搬运模式，[pypto_pro.language.AccToVecMode](../basic_data_structures/AccToVecMode.md)类型。<br>- DualModeSplitN与phase同时使用时，若N未32对齐，可能出现卡死现象。可以在矩阵乘运算前对L0B Buffer的N设置valid shape，使其向上对齐到32的倍数。 |
| relu_pre_mode | 输入 | 可选，L0C Buffer→UB或L1 Buffer搬运时是否开启随路ReLU操作，[pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md)类型。 |
| scale | 输入 | 可选，是否使能量化功能及设置量化模式下的量化参数，数据在搬出L0C Buffer时由Fixpipe乘以该比例并转换到目的数据类型。不支持与双目标搬运（AccToVecMode.DualModeSplitM / AccToVecMode.DualModeSplitN）同时使用。不同的传入形式会影响量化粒度，支持如下类型：<br>- **float类型**：直接传入固定值（如scale = 2.0），适用于整块Tile使用同一比例。<br>- **Scalar类型**：量化比例在运行时确定，需按数据类型传值。<br>&nbsp;&nbsp;- DT_FP32：直接传原始比例值（如0.5）。<br>&nbsp;&nbsp;- DT_INT32、DT_INT64：传预编码的float32位模式转成的整数（如struct.pack("!f", 0.5)）。<br>- **Tile类型**：每列使用独立比例，需满足以下要求：<br>&nbsp;&nbsp;- 目标存储区域必须为Fixpipe Buffer。<br>&nbsp;&nbsp;- shape为[1, N]（列量化），N必须是16的倍数且N ≤ 512。<br>&nbsp;&nbsp;- dtype为DT_INT64。<br>&nbsp;&nbsp;- 目的操作数的Tile数据类型为DT_INT8时，Fixpipe Buffer中的Tile每个DT_INT64元素的bit46需置1，用于选择有符号量化；未置位时L0C Buffer中的负值会被按无符号解读。<br>&nbsp;&nbsp;- 用户需要先把比例数据从GM搬到L1 Buffer，再搬到Fixpipe Buffer，并完成MTE1→FIX同步。|
| phase | 输入 | 可选，L0C Buffer搬运时是否启用unit_flag同步，详见[phase使用约束](../cube_computation/phase.md)。 |

## 约束说明

- 数据类型及分形约束：

  | 源 → 目的 | 分形要求 | 数据类型要求 |
  |---|---|---|
  | L1 Buffer → L0A Buffer | 源支持ND、NZ、ZN、ZZ，目的固定为NZ。 | 源与目的必须相同，支持DT_INT8、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP16、DT_BF16、DT_FP32、DT_FP4E2M1、DT_FP4E1M2、DT_FP8E8M0。 |
  | L1 Buffer → L0B Buffer | 源支持ND、NZ、ZN、ZZ，目的固定为ZN。 | 源与目的必须相同，支持DT_INT8、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP16、DT_BF16、DT_FP32、DT_FP4E2M1、DT_FP4E1M2、DT_FP8E8M0。 |
  | UB → UB（目的Tile的shape不大于源Tile） | 不校验分形。 | 源与目的必须相同。<br>- 非ND → NZ：支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。<br>- ND → NZ：支持DT_INT8、DT_UINT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | UB → UB（目的Tile的shape大于源Tile） | ND → ND、NZ → NZ。 | 源与目的必须相同，支持DT_INT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | UB → L1 Buffer（目的Tile的shape不大于源Tile） | 源支持ND、NZ，目的不校验分形。 | 源与目的必须相同，支持DT_INT8、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP16、DT_BF16、DT_FP32、DT_FP4E2M1、DT_FP4E1M2、DT_FP8E8M0。 |
  | UB → L1 Buffer（目的Tile的shape大于源Tile） | 源支持ND、NZ，目的不校验分形。 | 源与目的必须相同，支持DT_INT8、DT_INT32、DT_FP16、DT_BF16、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | L1 Buffer → BiasTable Buffer | 目的layout固定为ND。 | 支持DT_INT32 → DT_INT32、DT_FP32 → DT_FP32、DT_FP16 → DT_FP32、DT_BF16 → DT_FP32。 |
  | L1 Buffer → Fixpipe Buffer | 不校验分形。 | 目的必须为DT_INT64或DT_UINT64，源数据类型不做限制。 |
  | L1 Buffer → L0A_MX Buffer | 源与目的分形均为ZZ。 | 源与目的必须相同，仅支持DT_FP8E8M0。 |
  | L1 Buffer → L0B_MX Buffer | 源与目的分形均为NN。 | 源与目的必须相同，仅支持DT_FP8E8M0。 |
  | L0C Buffer → UB（不配置scale） | NZ → ND，NZ → DN，NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
  | L0C Buffer → UB（配置scale） | NZ → ND，NZ → DN，NZ → NZ。 | 支持DT_FP32 → DT_INT8/DT_UINT8/DT_HF8/DT_FP16/DT_BF16/DT_FP8E4M3FN/DT_FP32，以及DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16。 |
  | L0C Buffer → L1 Buffer（源与目的Tile的shape相同，不配置scale） | NZ → NZ、NZ → ND、NZ → DN。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
  | L0C Buffer → L1 Buffer（源与目的Tile的shape相同，配置scale） | NZ → NZ、NZ → ND、NZ → DN。 | 支持DT_FP32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16/DT_HF8/DT_FP8E4M3FN，以及DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16。 |
  | L0C Buffer → L1 Buffer（源与目的Tile的shape不同，不配置scale） | NZ → NZ。 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
  | L0C Buffer → L1 Buffer（源与目的Tile的shape不同，配置scale） | NZ → NZ。 | 支持DT_FP32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16/DT_HF8/DT_FP8E4M3FN，以及DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16。 |

- 尾块场景下，需要搭配pypto_pro.language.set_validshape与[pypto_pro.language.TileType](../basic_data_structures/TileType.md)中的compact参数使用，否则可能出现精度失败或卡死现象。
- L1 Buffer → L0A_MX Buffer/L0B_MX Buffer要求目的Tile必须满足L0A_MX Buffer地址 = L0A Buffer地址 >> 4或L0B_MX Buffer地址 = L0B Buffer地址 >> 4，否则MX矩阵乘时会读取错误的量化系数。

## 返回值说明

无。

## 调用示例

### L1->L0A/L0B

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 32], pl.DT_FP16],
    out: pl.Tensor[[64, 32], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)      # L1 -> L0A
        pl.move(br, cur_b)      # L1 -> L0B
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    a = torch.randn([64, 128], device=device, dtype=torch.float16)
    b = torch.randn([128, 32], device=device, dtype=torch.float16)
    out = torch.zeros([64, 32], device=device, dtype=torch.float32)

    kernel(a, b, out)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### UB数据转置写入L1 Buffer

当左矩阵或右矩阵由Vector计算在UB中产生时，可先通过pypto_pro.language.move将ND转换为NZ，再将结果写入L1 Buffer：

- 不转置时，L1 Buffer Tile使用NZ，shape与UB中的NZ Tile相同。
- 转置时，L1 Buffer Tile使用ZN，shape的两个维度与UB中的NZ Tile互换。源Tile的NZ [R, C]与目的Tile的ZN [C, R]表示相同的物理分形。

| 矩阵 | 是否转置 | UB中的NZ Tile | L1 Buffer Tile |
|---|---|---|---|
| 左矩阵A[M, K] | 否 | shape=[M, K] | shape=[M, K]，NZ |
| 左矩阵A[M, K] | 是 | shape=[K, M] | shape=[M, K]，ZN |
| 右矩阵B[K, N] | 否 | shape=[K, N] | shape=[K, N]，NZ |
| 右矩阵B[K, N] | 是 | shape=[N, K] | shape=[K, N]，ZN |

分块写入L1 Buffer时，可通过offset指定写入位置；源Tile的有效区域必须完整落入目的Tile范围内。

以下示例中，vector_result表示Vector计算产生的[K, M]数据。第一次move将其转换为NZ，第二次move将其转置写入L1 Buffer，得到供左矩阵使用的[M, K]、ZN数据。

```python
import pypto_pro.language as pl

M, K = 64, 128

# 前序Vector计算的输出：UB中的ND [K, M]
vector_result = pl.make_tile(
    pl.TileType(
        shape=[K, M], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, layout=pl.ND,
    ),
    addr=0x0000,
)

# ND转换为NZ时使用的UB Tile
vector_nz = pl.make_tile(
    pl.TileType(
        shape=[K, M], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
    ),
    addr=0x4000,
)

# 转置后的数据写入L1 Buffer，逻辑shape为[M, K]
lhs_l1 = pl.make_tile(
    pl.TileType(
        shape=[M, K], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
    ),
    addr=0x20000,
)

with pl.section_vector():
    pl.move(vector_nz, vector_result)  # ND [K, M] → NZ [K, M]
    pl.move(lhs_l1, vector_nz)         # NZ [K, M] → ZN [M, K]
```

### 量化参数的使用

```python
# 方式一：scale为编译期常量
# acc:      L0C Buffer中的Tile,   DT_FP32
# vec_tile: UB Tile,    DT_INT8
pl.matmul(acc, q_left, k_right)
pl.move(vec_tile, acc, scale=0.5)        # L0C Buffer -> UB，随路按 0.5 量化为 INT8


# 方式二：scale为运行时标量
# 入参声明为DT_INT32、DT_INT64时，需先把比例编码成float32位模式，当前示例。
# 入参声明为DT_FP32时，直接传比例数值本身，无需编码。
import struct

@pl.jit()
def kernel(..., scale_bits: pl.DT_INT32):
    # ...
    pl.move(vec_tile, acc, scale=scale_bits)

scale_bits = struct.unpack("!I", struct.pack("!f", 0.5))[0]
kernel(..., scale_bits=scale_bits)


# 方式三：scale为Tile类型
# fp_mat:   L1 Buffer中的Tile,      shape [1, 64], DT_INT64
# fp_tile:  Fixpipe Buffer中的Tile, shape [1, 64], DT_INT64
# acc:      L0C Buffer中的Tile,                    DT_INT32
# vec_tile: UB Tile,                     DT_FP16
@pl.jit()
def kernel(..., fp_params: pl.Tensor[[1, 64], pl.DT_INT64]):
    # ...
    pl.load(fp_mat, fp_params, [0, 0])       # GM -> L1
    pl.move(fp_tile, fp_mat)                 # L1 Buffer -> Fixpipe Buffer
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

    pl.matmul(acc, q_left, k_right)
    pl.move(vec_tile, acc, scale=fp_tile)    # L0C Buffer -> UB，按列独立比例反量化为DT_FP16

# 情况一、Kernel需要DT_INT64，若使用float32的比例张量，需要先进行转换
import torch_npu

scale_value = 2.0
scale_fp32 = torch.ones(1, 64, dtype=torch.float32, device=device) * scale_value
fp_params = torch_npu.npu_trans_quant_param(scale_fp32)

# 情况二、目的Tile数据类型为DT_INT8时，需要将Fixpipe Buffer中的Tile每个INT64元素的bit46需置1
def _make_scale_tensor(device: str, scale_values: list) -> torch.Tensor:
    scale_bits_list = []
    for scale_value in scale_values:
        scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]
        scale_bits |= 1 << 46  # signed INT8 flag
        scale_bits_list.append(scale_bits)
    return torch.tensor(scale_bits_list, dtype=torch.int64, device=device).reshape(1, 64)

scale_values = [2.0] * 64
fp_params = _make_scale_tensor(device, scale_values)
```
