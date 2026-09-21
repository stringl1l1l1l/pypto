# pypto.quant_mx

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

将1-4维ND格式的高精度浮点Tensor量化为MX（Microscaling）格式，返回量化结果和共享指数scale。

- 输入Tensor支持DT_FP16、DT_BF16、DT_FP32。
- 输出量化Tensor支持DT_FP8E4M3、DT_FP4_E2M1X2。其中DT_FP4_E2M1X2仅支持DT_FP16、DT_BF16输入。
- scale Tensor的数据类型固定为DT_FP8E8M0。
- 支持对尾轴（`axis=-1`）或次尾轴（`axis=-2`）进行量化，支持ROUND_DOWN（OCP）和ROUND_UP（NV）模式。
- `axis=-1`支持性能模式和非性能模式。非性能模式支持更灵活的view shape和TileShape设置，有利于算子融合，但单算子性能有所下降。

若输入shape记为 $[d_0, d_1, ..., d_{n-1}]$，则：

- 量化结果`quantized`的shape与`input`相同。
- `axis=-1`时，scale的shape为 $[d_0, d_1, ..., d_{n-2}, \lceil d_{n-1} / 64 \rceil, 2]$。
- `axis=-2`时，量化粒度仍为32，即每32个次尾轴元素共享一个指数；每两个连续分组组成一个64元素块。内部raw exp shape为 $[d_0, d_1, ..., d_{n-3}, d_{n-2}/64, d_{n-1}*2]$，对外返回的scale shape为 $[d_0, d_1, ..., d_{n-3}, d_{n-2}/64, d_{n-1}, 2]$，最后一维依次保存两个32元素分组的指数。

## 函数原型

```python
quant_mx(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_FP8E4M3,
    mode: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_DOWN,
    axis: int = -1,
    performance_mode: bool = True,
) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| input | 输入 | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16、DT_BF16、DT_FP32。<br>仅支持TILEOP_ND格式；Shape仅支持1-4维。 |
| quant_dtype | 输入 | 量化后输出Tensor的数据类型。<br>支持：DT_FP8E4M3、DT_FP4_E2M1X2。DT_FP4_E2M1X2仅支持DT_FP16、DT_BF16输入。 |
| mode | 输入 | 量化时共享指数的舍入模式。<br>支持：ROUND_DOWN（OCP）、ROUND_UP（NV）。 |
| axis | 输入 | 指定量化轴。<br>支持最后一维（`-1`或`input.shape.size() - 1`）和次尾轴（`-2`或`input.shape.size() - 2`）。 |
| performance_mode | 输入 | 是否启用性能模式。<br>默认值为`True`。 |

## 返回值说明

返回一个二元组`(quantized, scale)`：

- `quantized`：量化后的Tensor，数据类型由`quant_dtype`指定，Shape与`input`相同。
- `scale`：共享指数Tensor，数据类型固定为DT_FP8E8M0。`axis=-1`时Shape为`[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]`；`axis=-2`时Shape为`[*input.shape[:-2], input.shape[-2] / 64, input.shape[-1], 2]`。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
2. 若设置TileShape，其维度必须与输入一致。
3. `axis=-1`且`performance_mode=True`时，view shape尾轴必须满足256字节对齐并能整切运行时shape尾轴，且TileShape尾轴必须等于view shape尾轴。
4. `axis=-1`且`performance_mode=False`时，输入尾轴必须是64的倍数；TileShape尾轴只需为正数，无需等于view shape尾轴或满足256字节对齐。
5. `axis=-2`要求输入至少为二维。view shape、运行时有效shape和TileShape的次尾轴必须为正数且是64的倍数，不支持次尾轴尾块。
6. `axis=-2`且`quant_dtype=DT_FP8E4M3`时，view shape和TileShape的尾轴只需为正数，不要求256字节对齐，且TileShape尾轴无需等于view shape尾轴。
7. `axis=-2`且`quant_dtype=DT_FP4_E2M1X2`时，view shape和TileShape的尾轴必须为64的倍数，即打包后每行满足32字节对齐。
8. `axis=-2`时，`performance_mode=True`和`performance_mode=False`的约束相同。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape的具体取值要求参见[约束说明](#约束说明)。

示例1：`axis=-1`且`performance_mode=True`时，输入view shape为`[m, n]`，TileShape可设置为`[m1, n]`。

```python
pypto.set_vec_tile_shapes(4, 64)
```

示例2：`axis=-1`且`performance_mode=False`时，输入shape为`[m, n]`，TileShape可设置为`[m1, n1]`。

```python
pypto.set_vec_tile_shapes(2, 128)
```

### 接口调用示例

```python
x = pypto.tensor([8, 64], pypto.DT_FP32)

# 默认配置：DT_FP8E4M3 + ROUND_DOWN + 最后一维量化
quantized, scale = pypto.quant_mx(x)

# 显式指定OCP参数
quantized_perf, scale_perf = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_DOWN,
    -1,
    True,
)

# 使用NV scale算法
quantized_nv, scale_nv = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    True,
)

# 关闭性能模式
x_non_perf = pypto.tensor([2, 512], pypto.DT_FP32)
quantized_general, scale_general = pypto.quant_mx(
    x_non_perf,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    False,
)

# 对次尾轴进行量化
x_axis2 = pypto.tensor([128, 96], pypto.DT_FP16)
pypto.set_vec_tile_shapes(64, 96)
quantized_axis2, scale_axis2 = pypto.quant_mx(
    x_axis2,
    pypto.DT_FP8E4M3,
    pypto.ROUND_DOWN,
    -2,
    False,
)
# quantized_axis2.shape为[128, 96]，scale_axis2.shape为[2, 96, 2]
```

结果示例如下：

```python
Input x.shape: [8, 64]
Input x.dtype: DT_FP32
Output quantized.shape: [8, 64]
Output quantized.dtype: DT_FP8E4M3
Output scale.shape: [8, 1, 2]
Output scale.dtype: DT_FP8E8M0
```
