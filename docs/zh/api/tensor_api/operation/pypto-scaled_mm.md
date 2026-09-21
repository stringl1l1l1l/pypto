# pypto.scaled\_mm

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

实现mat_a 、mat_b矩阵的mx量化矩阵乘运算，计算公式为：out = (mat_a \* scale_a) @ (mat_b \* scale_b)

- mat_a 、mat_b 、scale_a 、scale_b为源操作数，mat_a为左矩阵；mat_b为右矩阵；scale_a为左矩阵量化参数；scale_b为右矩阵量化参数。
- out为目的操作数，存放矩阵乘结果的矩阵。

## 函数原型

```python
scaled_mm(mat_a, mat_b, out_dtype, scale_a, scale_b, *, a_trans = False, b_trans = False, scale_a_trans = False, scale_b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## 参数说明

表1：API参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| mat_a             | 输入      | 表示输入左矩阵。不支持输入空Tensor。<br> **数据类型**：详见表3。<br> **矩阵维度**：2维、3维、4维。<br> **Format**：TILEOP_ND，TILEOP_NZ（DT_FP8E5M2输入不支持TILEOP_NZ格式）。<br> **内轴外轴**：当输入矩阵mat_a非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K；当输入矩阵mat_a转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M。<br> **对齐要求**：当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。<br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，外轴16元素对齐。<br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐，外轴16元素对齐。 |
| mat_b              | 输入      | 表示输入右矩阵。不支持输入空Tensor。<br> **数据类型**：详见表3。<br> **矩阵维度**：2维、3维、4维。<br> **Format**：TILEOP_ND，TILEOP_NZ（DT_FP8E5M2输入不支持TILEOP_NZ格式）。<br> **内轴外轴**：当输入矩阵mat_b非转置时，对应数据排布为[K, N]，此时外轴为K，内轴为N；当输入矩阵mat_b转置时，对应数据排布为[N, K]，此时外轴为N，内轴为K。<br> **对齐要求**：当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。<br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，外轴16元素对齐。<br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐，外轴16元素对齐。 |
| out_dtype         | 输出      | 表示输出矩阵数据类型。基础场景输出数据类型支持情况详见表3，量化场景输出数据类型支持情况详见表4。|
| scale_a              | 输入      | 表示输入左矩阵量化参数。不支持输入空Tensor。<br> **数据类型**：详见表3。<br> **量化参数维度**：比mat_a多1维，mat_a为2/3/4维时，scale_a分别为3/4/5维。<br> **Format**：TILEOP_ND。<br> **量化参数shape**：记Ks=CeilAlign(K, 64)/64。二维场景非转置为[M, Ks, 2]，转置为[Ks, M, 2]；三维、四维场景在上述shape前增加mat_a自身的Batch前缀。scale_a的每个Batch轴必须与mat_a严格相等，并与mat_a使用相同的广播索引。|
| scale_b              | 输入      | 表示输入右矩阵量化参数。不支持输入空Tensor。<br> **数据类型**：详见表3。<br> **量化参数维度**：比mat_b多1维，mat_b为2/3/4维时，scale_b分别为3/4/5维。<br> **Format**：TILEOP_ND。<br> **量化参数shape**：记Ks=CeilAlign(K, 64)/64。二维场景非转置为[Ks, N, 2]，转置为[N, Ks, 2]；三维、四维场景在上述shape前增加mat_b自身的Batch前缀。scale_b的每个Batch轴必须与mat_b严格相等，并与mat_b使用相同的广播索引。|
| a_trans           | 输入      | 参数a_trans表示输入左矩阵是否转置，默认为False。 |
| b_trans           | 输入      | 参数b_trans表示输入右矩阵是否转置，默认为False。 |
| scale_a_trans     | 输入      | 参数scale_a_trans表示输入左矩阵量化参数是否转置，默认为False。 |
| scale_b_trans     | 输入      | 参数scale_b_trans表示输入右矩阵量化参数是否转置，默认为False。 |
| c_matrix_nz       | 输入      | 参数c_matrix_nz表示输出矩阵的Format是否采用NZ格式，默认为False，当前仅支持设置False，即输出矩阵仅支持ND格式。 |
| extend_params     | 输入      | 支持bias及fixpipe的量化功能，数据类型为字典格式。默认为None。详见表2 |

其中：
    - `CeilAlign(value, align)` 元素对齐实现为：`(value + align - 1) / align * align`

表2：extend_params参数说明

| 参数名            | 说明                                                                 |
|-------------------|----------------------------------------------------------------------|
| scale             | 表示pertensor量化场景（使用同一个缩放因子将高精度数映射到低精度数）输出矩阵量化的参数。<br>输入为float类型，取1位符号位 + 8位指数位 + 10位尾数位参与运算。<br>输入输出数据类型支持情况详见表4。<br>不支持叠加多核切k功能。|
| scale_tensor      | 表示perchannel量化场景（对每一个输出通道独立计算一套量化参数）输出矩阵量化的矩阵。<br>scale_tensor输入固定为uint64_t或int64_t的Tensor。计算时会转换64bit为float类型的低32bit后，取1位符号位 + 8位指数位 + 10位尾数位参与运算。<br>输入输出数据类型支持情况详见表4。<br>scale_tensor的第一维度必须置1，且N维度需要与mat_b矩阵的N维度相等。<br>scale_tensor只支持ND格式。<br>仅支持矩阵维度为2维场景。<br>不支持叠加多核切k功能。<br>量化输出类型为DT_INT8场景时，需要在function外提前调用torch_npu.npu_trans_quant_param并传入float32类型的torch.tensor来获取int64数据类型的scale_tensor。|
| bias_tensor       | 表示偏置矩阵。<br>输入为Tensor类型。<br>Bias矩阵数据类型可选DT_FP16、DT_BF16和DT_FP32。<br>bias_tensor只支持ND格式。<br>仅支持矩阵维度为2/3/4维场景。<br>当输入矩阵为3维时，Bias维度可以为[B, 1, N]或[1, N]，且N维度需要与mat_b矩阵的N维度相等。<br>当输入矩阵为4维时，Bias维度只能为[1, N]，且N维度需要与mat_b矩阵的N维度相等。<br>不支持叠加多核切K功能。|
| relu_type         | 表示输出矩阵是否进行ReLu操作。<br>输入为[ReLuType](../datatype/ReLuType.md)类型。<br>支持RELU和NO_RELU两种模式。<br>不支持叠加多核切k功能。 |

表3：scaled_mm基础场景支持的数据类型

| mat_a | mat_b | out_dtype | scale_a | scale_b | bias_tensor |
|:------|:------|:----------|:--------|:--------|:------------|
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 |
| DT_FP4_E2M1 | DT_FP4_E2M1 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 |

表4：scaled_mm量化场景支持的数据类型

| mat_a | mat_b | out_dtype |
|:------|:-----|:----------|
| DT_FP8E5M2 | DT_FP8E5M2，DT_FP8E4M3 | DT_INT8 |
| DT_FP8E4M3 | DT_FP8E5M2，DT_FP8E4M3 | DT_INT8 |
| DT_FP4_E2M1 | DT_FP4_E2M1 | DT_INT8 |

## 返回值说明

返回值为out矩阵（Tensor）。

## 约束说明

- 当输入为DT_FP4_E2M1量化场景时，需保证内轴为偶数。
- 调用scaled_mm接口前需要通过pypto.set\_cube\_tile\_shapes设置M、K、N轴上的切分大小。
- 调用scaled_mm接口的输入为调用pypto.reshape后的NZ格式时，需要调用pypto.set\_matrix\_size接口设置pypto.reshape前的输入到matmul的原始Shape的m,k,n值。

## 调用示例

```python
# 基本矩阵乘
mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([64, 2, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([2, 32, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# 3维单侧广播：A/scale_a作为整体沿Batch轴广播
mat_a = pypto.tensor([1, 128, 256], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([4, 256, 64], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([1, 128, 4, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([4, 4, 64, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# 4维交叉广播：scale Batch前缀分别与配对矩阵保持一致
mat_a = pypto.tensor([2, 1, 128, 256], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([1, 3, 256, 64], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 1, 128, 4, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([1, 3, 4, 64, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# 叠加Bias
mat_a = pypto.tensor([128, 64], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([32, 128], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
bias = pypto.tensor((1, 32), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True, extend_params=extend_params)

# 量化叠加RELU
scale_cpu = pypto.tensor((1, 32), pypto.DT_UINT64, "tensor_scale")
scale_tensor = torch_npu.npu_trans_quant_param(scale_cpu.npu()) # 生成scale_tensor
mat_a = pypto.tensor([128, 64], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([32, 128], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True, extend_params=extend_params)

```
