# pypto_pro.language.TensorLayout

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

数据布局枚举，用于描述GM Tensor的存储形式和Tile的数据排列形式。

- **GM Tensor**支持ND、NZ，默认ND。
- **Tile**支持ND、DN、NZ、ZN、NN、ZZ，默认值由内存空间决定。

## 原型定义

```python
PYPTO_DECLARE_ENUM(
    TensorLayout,
    ND,
    DN,
    NZ,
    ZN,
    NN,
    ZZ
)
```

## 参数说明

| 参数值 | 说明 |
|---|---|
| ND | 非分形行主序，最后一维连续。支持用于Tensor和Tile；GM Tensor默认使用ND，典型用于普通GM Tensor、UB Tile和Fixpipe Buffer。 |
| DN | 非分形列主序。仅支持用于Tile，典型用于UB上的[ROWS, 1]列向量，如归约结果和histogram索引。 |
| NZ | NZ分形排列。支持用于Tensor和Tile，典型用于GM Tensor、L1 Buffer、L0A Buffer和L0C Buffer。 |
| ZN | ZN分形排列。仅支持用于Tile，典型用于L0B Buffer和转置搬入时的L1 Buffer。 |
| ZZ | ZZ分形排列。仅支持用于Tile，用于MX矩阵乘中左量化系数矩阵在L1 Buffer和L0A_MX Buffer中的布局。 |
| NN | NN分形排列。仅支持用于Tile，用于MX矩阵乘中右量化系数矩阵在L1 Buffer和L0B_MX Buffer中的布局。 |

## 约束说明

- GM Tensor仅支持ND（行主序，默认）和NZ（分形布局）。

- NZ只声明GM内存的布局，不会把普通ND buffer自动转换成NZ。调用Kernel前，输入buffer必须已经按NZ物理顺序完成packing；NZ输出也必须使用按NZ格式分配的buffer。高维NZ Tensor的最后两轴固定解释为[M, N]，所有前导轴均作为batch轴，不支持在layout中指定任意分形轴。

- NZ将逻辑[..., M, N]存储为[..., ceil(N/C0), ceil(M/16), 16, C0]。M/N无需分形对齐，但实际存储空间必须按align(M, 16) × align(N, C0)的容量分配并使用上述NZ物理排布；仅分配M × N元素的紧凑buffer不受支持。补齐区不属于逻辑Tensor内容，框架不会为传入的buffer自动扩容或完成packing。DT_FP4E2M1和DT_FP4E1M2的C0为64；DT_INT8、DT_UINT8、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0和DT_HF8的C0为32；DT_FP16、DT_BF16、DT_INT16和DT_UINT16的C0为16；DT_FP32、DT_INT32和DT_UINT32的C0为8；DT_INT64和DT_UINT64的C0为4。4位数据类型的M/N同样按逻辑元素计数，不使用packed字节数作为Tensor shape。

- MX矩阵计算使用的E8M0分组缩放因子在GM中仍声明为普通ND Tensor；物理shape和搬运约束见[matmul_mx](../cube_computation/matmul_mx.md)和[load](../memory_data_movement/load.md)。

- Tile的布局约束请参见[pypto_pro.language.TileType](TileType.md)。

## 调用示例

### GM Tensor布局声明

```python
import pypto_pro.language as pl

x: pl.Tensor[[64, 128], pl.DT_FP16]                   # 默认ND
x_nz: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]         # 显式指定NZ
```

### UB Tile的ND与DN

```python
import pypto_pro.language as pl

# UB中的普通数据Tile使用ND
tile_src = pl.TileType(shape=[32, 128], dtype=pl.DT_UINT16,
                       target_memory=pl.MemorySpace.Vec, layout=pl.ND)

# UB中的单列Tile使用DN，例如归约结果
tile_red = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec, layout=pl.DN)
```

### Cube分形布局

Cube计算涉及的L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer采用不同的默认分形布局。以下示例显式写出各内存空间的默认布局：

```python
import pypto_pro.language as pl

# L1 Buffer和L0A Buffer默认使用NZ
mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ)

# L0B Buffer默认使用ZN
right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN)

# L0C Buffer默认使用NZ
acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
```

### MX矩阵乘量化系数的ZZ与NN

ZZ和NN分别用于MX矩阵乘的左、右量化系数矩阵。量化系数的数据类型为DT_FP8E8M0，逻辑shape分别为[M, K/32]和[K/32, N]。L1 Buffer默认使用NZ，因此需要为左、右量化系数Tile显式指定ZZ和NN；L0A_MX Buffer和L0B_MX Buffer则默认使用对应布局。

```python
import pypto_pro.language as pl

M, K, N = 64, 128, 64
G = K // 32

# L1 Buffer中的左、右量化系数Tile
scale_a_l1_type = pl.TileType(
    shape=[M, G], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
)
scale_b_l1_type = pl.TileType(
    shape=[G, N], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat, layout=pl.NN,
)

# L0A_MX Buffer和L0B_MX Buffer未指定layout时分别默认使用ZZ和NN
scale_a_type = pl.TileType(
    shape=[M, G], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleLeft,
)
scale_b_type = pl.TileType(
    shape=[G, N], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleRight,
)
```
