# pypto.conv

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

实现输入input_conv、weight完成卷积运算，支持bias参数，计算公式为：out = input_conv @ weight + bias (@表示为卷积处理)。

- input_conv、weight、bias为源操作数；input_conv为输入矩阵，weight为权重矩阵，bias为输入的偏置。
- out为目的操作数，存放卷积处理结果的矩阵。
- 当前暂不支持fixpipe量化场景。

## 函数原型

```python
conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups=1, transposed=False, output_paddings=[], extend_params=None) -> Tensor
```

## 参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input_conv       | 输入      | 输入特征图Tensor。<br>不支持空Tensor。<br>支持维度：3D（1D conv）、4D（2D conv）、5D（3D conv）。<br>支持格式：NCL、NCHW、NCDHW。<br>支持数据类型：DT_FP16、DT_BF16、DT_FP32。<br>shape约束：各维度取值范围 [1, 1000000]。input_conv的Cin需满足：weight的Cin * groups = input_conv的Cin。 |
| weight            | 输入      | 卷积核Tensor。<br>维度必须与input_conv一致（3D/4D/5D）。<br>数据类型必须与input_conv一致。<br>shape约束：各维度取值范围 [1, 1000000]。不同型号存在额外约束，详细请参见[约束说明](#约束说明)。 |
| out_dtype         | 输入      | 输出Tensor数据类型。<br>支持：DT_FP16、DT_BF16、DT_FP32。<br>必须与input_conv一致；fixpipe量化场景可单独指定。 |
| strides           | 输入      | 卷积步长，单向参数。<br>- 1D（1D conv）<br>- 2D（2D conv）<br>- 3D（3D conv）<br>取值范围：[1, 63]。 |
| paddings          | 输入      | 卷积填充，双向参数。<br>- 2D（1D conv）<br>- 4D（2D conv）<br>- 6D（3D conv）<br>取值范围：[0, 255]，且每维填充值 < 对应卷积核大小。 |
| dilations         | 输入      | 空洞卷积膨胀率，单向参数。<br>- 1D（1D conv）<br>- 2D（2D conv）<br>- 3D（3D conv）<br>取值范围：[1, 63]。 |
| groups            | 输入      | 分组卷积组数，默认1。<br>取值范围：[1, 65535]。<br>Cin、Cout必须可被groups整除。 |
| transposed        | 输入      | 是否为转置卷积（反卷积），默认False。<br>当前暂不支持True。 |
| output_paddings   | 输入      | 转置卷积输出端填充，仅transposed=True时使用。<br>当前暂不支持。 |
| extend_params     | 输入      | 扩展参数字典，支持bias_tensor、scale、relu_type、scale_tensor：<br>- bias_tensor：可选的偏置张量，形状为(Cout,)，仅支持ND格式，不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>- scale：浮点型，per-tensor缩放因子。<br>- scale_tensor：uint64类型，per-channel缩放Tensor，shape [1, Cout]，仅ND格式。<br>- relu_type：激活类型，支持RELU/NO_RELU等。 |

## 返回值说明

返回卷积运算后的输出Tensor：

- 1D卷积输出shape：(Batch, Cout, Wout)
- 2D卷积输出shape：(Batch, Cout, Hout, Wout)
- 3D卷积输出shape：(Batch, Cout, Dout, Hout, Wout)

输出shape各维度范围：[1, 1000000]。

## 约束说明

### Shape合法性约束

- 输入特征图（input_conv）：Batch、Cin、Hin、Win、Din维度必须在 [1, 1000000] 范围内；
- 卷积核（weight）：Cout、Kh、Kw、Kd维度必须在 [1, 1000000] 范围内；
- 偏置（bias_tensor）：shape必须等于 [Cout]，否则校验失败；
- 输出特征图：Hout、Wout、Dout维度必须在 [1, 1000000] 范围内。

### 属性参数合法性约束

- 基础维度匹配约束：
  - strides维度数必须与卷积维度匹配（2D conv长度=2，3D conv长度=3）；
  - dilations维度数必须与卷积维度匹配（2D conv长度=2，3D conv长度=3）；
  - paddings维度数必须为2×卷积维度（2D conv长度=4，3D conv长度=6）。
- 数值范围约束：
  - strides取值范围 [1, 63]；
  - dilations取值范围 [1, 63]；
  - paddings取值范围 [0, 255]，且每维填充值 < 对应卷积核维度大小（如padding_h < Kh、padding_w < Kw）；
  - groups取值范围 [1, 65535]。
- 卷积核约束：
  - Kh ≤ 255、Kw ≤ 255；
  - Kh × Kw × (32 / sizeof(dtype)) ≤ 65535；dtype为input_conv的数据类型所占字节数，如FP16是2，FP32是4等。
- 通道数约束：
  - Cin（输入通道数）必须能被groups整除；
  - Cout（输出通道数）必须能被groups整除；
  - CinFmap = CinWeight × groups。

### 缓存空间约束

- 调用conv接口前，必须通过pypto.set_conv_tile_shapes接口设置L1/L0层级的卷积TileShape切分大小。

### 功能支持约束

- transposed=True（转置卷积）暂不支持，调用会抛出RuntimeError；
- input_conv/weight仅支持DT_FP16、DT_BF16、DT_FP32数据类型，其他类型会抛出ValueError；
- input_conv与weight的维度必须一致（如input_conv为4D则weight也需为4D），否则抛出RuntimeError。

### 动态轴切分支持

卷积算子支持的动态轴切分维度如下：

| 维度       | 是否支持 | 切分方式                           | 说明                                                                 |
|:-----------|:--------:|:-----------------------------------|:---------------------------------------------------------------------|
| Batch      |    √     | 前端循环切分                        | TileL1Info.tileBatch必须为1（硬件约束），通过前端循环实现动态切分   |
| Cout       |    √     | TileShape动态切分 + 前端循环       | 支持TileShape配置的动态切分，配合前端循环实现完整Cout维度覆盖（当groups > 1时不允许切分）      |
| Dout       |    √     | TileShape动态切分 + 前端循环       | 仅3D卷积支持，Dout维度动态切分                                     |
| Hout       |    √     | TileShape动态切分 + 前端循环       | Hout维度动态切分，配合前端循环实现完整覆盖                           |
| Wout       |    √     | TileShape动态切分 + 前端循环       | Wout维度动态切分，配合前端循环实现完整覆盖                           |
| Cin        |    ×     | -       | Cin维度暂不支持动态轴切分，请使用set_conv_tile_shapes接口实现Cin和k的tile切分                   |

### 数据类型约束

<!-- npu="950" id9 -->
- Ascend 950PR&950DT系列产品：支持的数据类型为DT_FP16、DT_BF16、DT_FP32。input_conv、weight、bias和out的数据类型需要相同。
<!-- end id9 -->
<!-- npu="A3" id10 -->
- Atlas A3系列产品：支持的数据类型为DT_FP16、DT_BF16、DT_FP32。对于DT_FP16和DT_FP32类型，input_conv、weight、bias和out的数据类型需要相同；对于DT_BF16类型，input_conv、weight和out为DT_BF16类型，bias需为DT_FP32类型。
<!-- end id10 -->
<!-- npu="910b" id11 -->
- Atlas A2系列产品：支持的数据类型为DT_FP16、DT_BF16、DT_FP32。对于DT_FP16和DT_FP32类型，input_conv、weight、bias和out的数据类型需要相同；对于DT_BF16类型，input_conv、weight和out为DT_BF16类型，bias需为DT_FP32类型。
<!-- end id11 -->

## 调用示例

```python
# 2D卷积基础示例
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1])

# 2D卷积带bias和ReLu
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
extend_params = {'bias_tensor': bias, 'relu_type': pypto.ConvReLuType.RELU}

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1],
                   extend_params=extend_params)

# 3D卷积示例
input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1, 1],
                   paddings=[0, 0, 0, 0, 0, 0],
                   dilations=[1, 1, 1])
```
