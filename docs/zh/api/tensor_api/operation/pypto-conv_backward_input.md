# pypto.conv_backward_input

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

实现卷积反向输入梯度（conv backward input）计算，根据输出梯度grad_output和卷积核weight，计算输入特征图的梯度。

## 函数原型

```python
conv_backward_input(grad_output, input_size, weight, out_dtype, strides, paddings, dilations, *, groups=1) -> Tensor
```

## 参数说明

| 参数名         | 输入/输出 | 说明                                                                 |
|----------------|-----------|----------------------------------------------------------------------|
| grad_output    | 输入      | 输出梯度Tensor。<br>不支持空Tensor。<br>仅支持4D（2D conv），仅支持NCHW格式。<br>支持数据类型：DT_FP16、DT_BF16。<br>shape约束：各维度取值范围 [1, 1000000]。 |
| input_size     | 输入      | 输入特征图的shape，也是输出梯度的shape，维度和grad_output一致。 |
| weight         | 输入      | 卷积核Tensor。<br>维度和grad_output一致（4D），数据类型必须与grad_output一致。<br>shape约束：各维度取值范围 [1, 1000000]， Kh、Kw必须在 [1, 255] 范围内，且Kh × Kw × 32bytes/sizeof(dtype) ≤ 65535。 |
| out_dtype      | 输入      | 输出Tensor数据类型。<br>支持：DT_FP16、DT_BF16，且必须与grad_output数据类型一致。 |
| strides        | 输入      | 卷积步长，参数示例：[1, 1]。<br>各维度取值范围：[1, 63]。 |
| paddings       | 输入      | 卷积填充，参数示例：[1, 1, 2, 2]。<br>各维度取值范围：[0, 255]，且paddingH < (Kh-1) * dilationH + 1、paddingW < (Kw-1) * dilationW + 1。<br>当前仅支持双边相同pad，即padLeft == padRight、padTop == padBottom。 |
| dilations      | 输入      | 空洞卷积膨胀率，参数示例：[1, 1]。<br>各维度取值范围：[1, 63]。 |
| groups         | 输入      | 分组卷积组数，默认1。<br>当前仅支持groups = 1。 |

## 返回值说明

返回卷积反向输入梯度计算后的输出Tensor：

2D卷积反向输出shape：(Batch, Cin, Hin, Win)

输出shape各维度范围：[1, 1000000]。

**注意**：仅依赖grad_output/weight以及属性并不能推导出唯一的输出shape，由于shape计算过程中需要对stride做除法并向下取整导致，所以输出shape是一个范围。以下公式为每个轴的有效范围计算公式，在以下公式产生的[min, max]间的都是有效值，使用者需要将想要输出的shape传入input_size参数：

```txt
HinMin = (Hout - 1) * stride_h - pad_top - pad_bottom + (Kh - 1) * dilation_h + 1
WinMin = (Wout - 1) * stride_w - pad_left - pad_right + (Kw - 1) * dilation_w + 1
HinMax = HinMin + stride_h - 1
WinMax = WinMin + stride_w - 1
```

## 约束说明

- 调用conv_backward_input接口前，必须通过pypto.set_convbp_input_tile_shapes接口设置L1/L0层级的卷积反向TileShape切分大小;调用pypto.set_vec_tile_shapes接口设置输入shape每个维度TileShape切分大小。

- 不支持bias。

- 不支持动态shape。

## 调用示例

```python
# 2D卷积反向基础示例
grad_output = pypto.tensor((1, 16, 5, 32), pypto.DT_FP16, "grad_output")
weight = pypto.tensor((16, 16, 3, 3), pypto.DT_FP16, "weight")

# 计算输入梯度shape
hin = 5
win = 32
input_size = (1, 16, hin, win)
tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=144, tileNL1=16
    )
tile_l0_info = pypto_impl.ConvBpTileL0Info(
    tileML0=16, tileKL0=16, tileNL0=16
)
pypto.set_convbp_input_tile_shapes(tile_l1_info, tile_l0_info)
pypto.set_vec_tile_shapes(16, 16, 16, 16)
out = pypto.conv_backward_input(
    grad_output,
    input_size,
    weight,
    pypto.DT_FP16,
    strides=[1, 1],
    paddings=[1, 1, 1, 1],
    dilations=[1, 1]
)
```
