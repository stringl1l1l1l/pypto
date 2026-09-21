# pypto.set_cube_tile_shapes

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

在调用`pypto.matmul`或`pypto.scaled_mm`前必须调用本接口设置矩阵运算的切分大小，具体切分配置可参考[Matmul高性能编程](../../../guide/programming_guide/tensor/debug/matmul_performance_guide.md)。

## 函数原型

```python
set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], enable_split_k: bool = False) -> None
```

## 参数说明

| 参数名            | 输入/输出 | 说明                                                                                               |
|-------------------|-----------|--------------------------------------------------------------------------------------------------|
| m                      | 输入      | m维度在L0和L1上的TileShape（切片形状）的切分大小，为包含两个整数的列表，分别对应mL0和mL1的切分大小 |
| k                      | 输入      | k维度在L0和L1上的TileShape（切片形状）的切分大小，为包含两个整数的列表，分别对应kL0和kL1的切分大小 |
| n                      | 输入      | n维度在L0和L1上的TileShape（切片形状）的切分大小，为包含两个整数的列表，分别对应nL0和nL1的切分大小 |
| enable_split_k         | 输入      | 设置True表示使能matmul的多核切K功能（便捷开关，**不保证性能最优**）；默认为False，表示未使能多核切K。<br>性能调优推荐通过前端手动切K实现，详见[Matmul高性能编程](../../../guide/programming_guide/tensor/debug/matmul_performance_guide.md) |

## 返回值说明

void

## 约束说明

- 对齐约束

    - 通用对齐约束

    要求mL0、mL1、kL0、kL1、nL0、nL1均满足32字节对齐（DT_FP32输入场景要求满足16元素对齐）。例如：输入矩阵的数据类型为DT_FP16时，kL0 * sizeof(DT_FP16) % 32 == 0。

    - 基础关系约束

    | 约束项 | 要求 |
    |:-------|:-----|
    | mL0与mL1 | `mL0 > 0`且`mL0 ≤ mL1`且`mL1 % mL0 == 0` |
    | kL0与kL1 | `kL0 > 0`且`kL0 ≤ kL1`且`kL1 % kL0 == 0` |
    | nL0与nL1 | `nL0 > 0`且`nL0 ≤ nL1`且`nL1 % nL0 == 0` |

    - ND格式特有约束

    当A矩阵在format为ND且转置场景时（即数据排布为[K, M]），要求mL0满足32字节对齐。

    - NZ格式特有约束

    A、B矩阵在format为NZ场景时，要求外轴切分大小满足16元素对齐，内轴切分大小满足32字节对齐。例如，在A矩阵非转置场景，外轴为M、内轴为K，要求mL0、mL1满足16元素对齐，kL0、kL1满足32字节对齐。

    - scaled_mm特有约束

    当调用`pypto.scaled_mm`时，要求满足 `kL0 % 64 == 0`。

- 空间约束

    - 输入dtype为DT_FP16或DT_BF16或DT_FP32：

    ```txt
    CeilAlign(mL0, 16) × CeilAlign(kL0, 16) × sizeof(aDtype) ≤ L0A_size
    CeilAlign(nL0, 16) × CeilAlign(kL0, 16) × sizeof(bDtype) ≤ L0B_size
    CeilAlign(mL0, 16) × CeilAlign(nL0, 16) × sizeof(cDtype) ≤ L0C_size
    CeilAlign(mL1, 16) × CeilAlign(kL1, 16) × sizeof(aDtype) + CeilAlign(nL1, 16) × CeilAlign(kL1, 16) × sizeof(bDtype) ≤ L1_size
    ```

    - 输入dtype为DT_INT8或DT_FP8E5M2或DT_FP8E4M3或DT_HF8：

    ```txt
    CeilAlign(mL0, 32) × CeilAlign(kL0, 32) × sizeof(aDtype) ≤ L0A_size
    CeilAlign(nL0, 32) × CeilAlign(kL0, 32) × sizeof(bDtype) ≤ L0B_size
    CeilAlign(mL0, 32) × CeilAlign(nL0, 32) × sizeof(cDtype) ≤ L0C_size
    CeilAlign(mL1, 32) × CeilAlign(kL1, 32) × sizeof(aDtype) + CeilAlign(nL1, 32) × CeilAlign(kL1, 32) × sizeof(bDtype) ≤ L1_size
    ```

    - Bias空间约束：

    bias数据到达BTBuffer全部转为fp32，需满足以下约束：

    ```txt
    nL0 × 4 ≤ BTBuffer_size
    ```

    - FixPipe空间约束：

    scaleTensor数据为uint64_t，需满足以下约束：

    ```txt
    nL0 × 8 ≤ FixBuffer_size
    ```

    其中：
    - aDtype、bDtype为输入矩阵数据类型
    - cDtype为输出矩阵数据类型，当输入为DT_INT8时cDtype为DT_INT32，其余场景cDtype为DT_FP32
    - `CeilAlign(value, align)`元素对齐实现为：`(value + align - 1) / align * align`

- 多核切K约束

    - `pypto.matmul`支持2维/3维/4维矩阵多核切K。
    - `pypto.scaled_mm`仅支持2维矩阵多核切K，3维/4维矩阵不支持。
    - 多核切K场景只支持out\_dtype数据类型为DT\_FP32或DT\_INT32。
    - Bias/FixPipe(包含ReLU)场景不支持叠加多核切K功能。
    - 仅支持通过`pypto.view()`接口创建tensor的场景，且要求K轴的valid_shape与K轴shape完全相同，接口使用详见[pypto.view](../operation/pypto-view.md)。

## 调用示例

```python
# 基本配置
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 启用多核切K（便捷开关，不保证性能最优；性能调优见Matmul高性能编程）
m = 1024
n = 1024
k = 1024
a_tensor = pypto.tensor((m, k), pypto.DT_FP16, "a_tensor")
b_tensor = pypto.tensor((k, n), pypto.DT_FP16, "b_tensor")
m_view = 256
m_loop = m // m_view
pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128], enable_split_k=True)
for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
    m_offset = m_idx * m_view
    a_view = pypto.view(a_tensor, [m_view, k],
                                              [m_idx * m_view, 0],
                                              valid_shape=[(m - m_offset).min(m_view), k])
    pypto.matmul(a_view, b_tensor, pypto.DT_FP32)
```
