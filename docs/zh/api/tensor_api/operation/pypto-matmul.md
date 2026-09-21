# pypto.matmul

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

实现input 、mat2矩阵的矩阵乘运算，计算公式为：out = input @ mat2

- input 、mat2为源操作数，input为左矩阵；mat2为右矩阵。
- out为目的操作数，存放矩阵乘结果的矩阵。

## 注意事项

- **左右矩阵数据类型必须一致**：matmul的左右矩阵数据类型必须相同（如BF16+BF16、FP16+FP16），不支持混合输入（如BF16+FP32），FP8数据类型除外。
- **避免不必要的cast**：将BF16升级到FP32再进行matmul计算不会有精度提升，反而会产生额外的数据搬移开销。
- **利用随路transpose**：matmul支持`a_trans`和`b_trans`参数，可以在矩阵乘时随路完成转置，避免额外调用transpose操作。
- **必须先设置TileShape**：调用matmul接口前需要通过`set_cube_tile_shapes`设置M、K、N轴上的切分大小。

## 函数原型

```python
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## 参数说明

表1：API参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input             | 输入      | 表示输入左矩阵，不支持输入空Tensor。<br> **数据类型**：详见表3。<br> **矩阵维度**：2维、3维、4维，且左右矩阵维度需保持一致。<br> **Format**：TILEOP_ND，TILEOP_NZ（DT_FP32，DT_FP8E5M2，DT_HF8输入不支持TILEOP_NZ格式）。<br> **内轴外轴**：当输入矩阵input非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K；当输入矩阵input转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M；<br> **对齐要求**：当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。<br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，外轴16元素对齐。<br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐，外轴16元素对齐。|
| mat2              | 输入      | 表示输入右矩阵，不支持输入空Tensor。<br> **数据类型**：详见表3。<br> **矩阵维度**：2维、3维、4维，且左右矩阵维度需保持一致。<br> **Format**：TILEOP_ND，TILEOP_NZ（DT_FP32，DT_FP8E5M2，DT_HF8输入不支持TILEOP_NZ格式）。<br> **内轴外轴**：当输入矩阵mat2非转置时，对应数据排布为[K, N]，此时外轴为K，内轴为N；当输入矩阵mat2转置时，对应数据排布为[N, K]，此时外轴为N，内轴为K；<br> **对齐要求**：当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。<br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，外轴16元素对齐。<br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐，外轴16元素对齐。 |
| out_dtype         | 输出      | 表示输出矩阵数据类型。基础场景输出数据类型支持情况详见表3，反量化及量化场景输出数据类型支持情况详见表4，表5。|
| a_trans           | 输入      | 参数a_trans表示输入左矩阵是否转置，默认为False。 |
| b_trans           | 输入      | 参数b_trans表示输入右矩阵是否转置，默认为False。 |
| c_matrix_nz       | 输入      | 参数c_matrix_nz表示输出矩阵的Format是否采用NZ格式，默认为False，当前仅支持设置False，即输出矩阵仅支持ND格式。 |
| extend_params     | 输入      | 支持bias、fixpipe量化和反量化及TF32舍入模式功能，详见表2。<br>bias、fixpipe量化和反量化输入输出数据类型支持情况详见表3，表4，表5。<br>- 数据类型为字典格式。<br>- 此参数与其内部参数均为可选参数。|

表2：extend_params参数说明

| 参数名            | 说明                                                                 |
|-------------------|----------------------------------------------------------------------|
| scale             | 表示per-tensor量化场景（使用同一个缩放因子将高精度数映射到低精度数）输出矩阵反量化的参数。<br>输入为float类型，取1位符号位 + 8位指数位 + 10位尾数位参与运算。<br>输入输出数据类型支持情况详见表4，表5。<br>不支持叠加多核切k功能。|
| scale_tensor      | 表示per-channel量化场景（对每一个输出通道独立计算一套量化参数）输出矩阵反量化的矩阵。<br>scale_tensor输入固定为uint64_t或int64_t的Tensor。计算时会转换64位bit为float类型的低32位bit后，取1位符号位 + 8位指数位 + 10位尾数位参与运算。<br>输入输出数据类型支持情况详见表4、表5。<br>scale_tensor的倒数第二维度的形状必须置1，且N维度需要与mat2矩阵的N维度相等。<br>scale_tensor只支持ND格式。<br>不支持叠加多核切k功能。<br>量化输出类型为DT_INT8场景时，需要提前调用torch_npu.npu_trans_quant_param并传入float32类型的torch.tensor来获取int64数据类型的scale_tensor。|
| bias_tensor       | 表示偏置矩阵。<br>输入为Tensor类型。<br>输入输出数据类型支持情况详见表3。<br>bias_tensor只支持ND格式。<br>bias_tensor的倒数第二维度的形状应置1，且N维度需要与mat2矩阵的N维度相等。<br>矩阵维度为4维场景下，bias仅允许2维输入。<br>不支持叠加多核切k功能。 |
| relu_type         | 表示输出矩阵是否进行ReLu操作。<br>输入为[ReLuType](../datatype/ReLuType.md)类型。<br>支持RELU和NO_RELU两种模式。<br>不支持叠加多核切k功能。|
| trans_mode        | 表示是否使能TF32计算及TF32舍入模式。<br>输入为[TransMode](../datatype/TransMode.md)类型，支持以下三种模式：<br>• CAST_NONE：不使能float数据类型转换为TF32数据类型。<br>• CAST_RINT：使能float数据类型转换为TF32数据类型，舍入规则：舍入到最近整数，中间值时舍入到偶数。<br>• CAST_ROUND：使能float数据类型转换为TF32数据类型，舍入规则：舍入到最近整数，中间值时远离零舍入。<br>仅支持输入左右矩阵和输出矩阵数据类型均为DT_FP32时设置。 |

表3：Matmul基础场景支持的数据类型

| input | mat2 | out_dtype | bias_tensor |
|:------|:-----|:----------|:------------|
| DT_FP16 | DT_FP16 | DT_FP16/DT_FP32 | DT_FP16/DT_FP32 |
| DT_BF16 | DT_BF16 | DT_BF16/DT_FP32 | DT_BF16/DT_FP32 |
| DT_FP32 | DT_FP32 | DT_FP32 | DT_FP32 |
| DT_INT8 | DT_INT8 | DT_INT32 | DT_INT32 |
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 |
| DT_HF8 | DT_HF8 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 |

表4：Matmul反量化场景支持的数据类型

| input | mat2 | out_dtype |
|:------|:-----|:----------|
| DT_INT8 | DT_INT8 | DT_FP16 |

表5：Matmul量化场景支持的数据类型

| input | mat2 | out_dtype |
|:------|:-----|:----------|
| DT_BF16 | DT_BF16 | DT_INT8 |
| DT_FP16 | DT_FP16 | DT_INT8 |
| DT_FP32 | DT_FP32 | DT_INT8 |
| DT_INT8 | DT_INT8 | DT_INT8 |
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_INT8 |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_INT8 |
| DT_HF8 | DT_HF8 | DT_INT8 |

## 返回值说明

返回值为out矩阵（Tensor）。

## 约束说明

<!-- npu="910b" id4 -->
- Atlas A2系列产品：不支持DT_HF8，DT_FP8E5M2，DT_FP8E4M3，不支持extend_params中的trans_mode参数；当输入矩阵的数据类型为DT_BF16时，bias_tensor数据类型不支持DT_BF16。
<!-- end id4 -->
<!-- npu="A3" id5 -->
- Atlas A3系列产品：不支持DT_HF8，DT_FP8E5M2，DT_FP8E4M3，不支持extend_params中的trans_mode参数。当输入矩阵的数据类型为DT_BF16时，bias_tensor数据类型不支持DT_BF16。
<!-- end id5 -->
- 调用matmul接口前需要通过pypto.set\_cube\_tile\_shapes设置M、K、N轴上的切分大小
- 当矩阵维度为3维或者4维时，需要调用pypto.set\_vec\_tile\_shapes接口设置vector的TileShape切分，如未设置，接口内部会设置2维的vec\_tile\_shape，其值为128，128。
- 调用matmul接口的输入为调用pypto.reshape后的NZ格式时，需要调用pypto.set\_matrix\_size接口设置pypto.reshape前的输入到matmul的原始Shape的m，k，n值。
- 调用matmul接口的输入矩阵维度为3维/4维并且数据格式为NZ格式时，需要调用pypto.set\_matrix\_size接口设置输入到matmul的原始Shape的m，k，n值。

## 调用示例

```python
# 基本矩阵乘
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a1 = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
b1 = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")
out1 = pypto.matmul(a1, b1, pypto.DT_BF16)

# 批量矩阵乘
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a2 = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
b2 = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
out2 = pypto.matmul(a2, b2, pypto.DT_FP16)

# 批次广播
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a3 = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b3 = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
out3 = pypto.matmul(a3, b3, pypto.DT_FP32)

# 叠加Bias
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)

# 反量化
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

# 反量化叠加RELU
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

# 反量化叠加RELU
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

# 量化
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_INT8, extend_params=extend_params)

# 量化叠加RELU
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
scale_cpu = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
scale_tensor = torch_npu.npu_trans_quant_param(scale_cpu.npu()) # 生成scale_tensor
a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_INT8, extend_params=extend_params)
```

<!-- npu="950" id6 -->
```python
# TF32计算模式（Ascend 950PR&950DT系列产品）
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
extend_params = {'trans_mode': pypto.TransMode.CAST_ROUND}
pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
```
<!-- end id6 -->
