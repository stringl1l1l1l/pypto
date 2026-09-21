# pypto_pro.language.cast

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

对源Tile有效区域内的元素逐元素进行数据类型转换，并将结果写入目标Tile的对应位置。目标数据类型由out.dtype确定：

$$
out_{i,j}=\operatorname{cast}_{mode}(src_{i,j})
$$

当转换会损失精度时，可通过mode指定舍入模式。输入值应处于目标数据类型的可表示范围内。

## 函数原型

```python
pypto_pro.language.cast(
    out: Tile,
    src: Tile,
    *,
    mode: RoundMode = pl.RoundMode.CAST_ROUND,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标操作数，须为UB Tile，目标数据类型由out.dtype确定。out.valid_shape指定本次转换的有效区域；src必须包含与该区域对应的有效元素，且源、目的Tile的形状和布局应满足逐元素对应访问要求。支持的数据类型组合见[支持的数据类型转换](#支持的数据类型转换)。调用完成后，out的有效区域被转换结果覆盖。 |
| src | 输入 | 源操作数，须为UB Tile。其有效区域必须覆盖out.valid_shape指定的转换区域，并保证该区域内的每个元素均可由out中的相同逻辑坐标访问。源、目的Tile的物理shape和行跨度可以不同，但均须为转换区域提供足够的存储空间。 |
| mode | 输入 | 可选，[pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md)枚举类型，必须是编译期可确定的RoundMode枚举值。默认值为pypto_pro.language.RoundMode.CAST_ROUND。该参数只在转换路径涉及舍入时影响结果；各枚举值语义见RoundMode，不同转换路径的支持情况见[舍入模式](#舍入模式)和[支持的数据类型转换](#支持的数据类型转换)。 |

## 约束说明

### 接口使用约束

- **目标数据类型不通过参数传入**：本接口没有dtype或target_type参数，目标数据类型由预先创建的out Tile确定。
- **接口不返回新Tile**：转换结果写入out，接口返回None。不能按Tensor接口的方式写成dst = pypto_pro.language.cast(src, dtype)。
- **同类型转换不是通用拷贝接口**：当前仅支持DT_FP32到DT_FP32的同类型转换，该路径执行舍入操作，并非原值拷贝。其他同类型组合不支持。

### 舍入模式

RoundMode各枚举值的通用语义参见[pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md)。本接口还存在以下与数据类型转换路径相关的规则：

- 对不涉及舍入的类型扩展或整数窄化路径，mode不参与结果计算，建议使用CAST_NONE表达意图。
- DT_FP16/DT_FP32 -> DT_HF8路径固定采用CAST_ROUND；传入其他mode不会改变该路径的舍入行为。
- 除DT_FP32 -> DT_FP16外，对其他路径传入CAST_ODD不会执行奇数舍入，而会回退到CAST_NONE对应的默认规则：浮点数转整数使用CAST_TRUNC，其他路径使用CAST_RINT。

### 支持的数据类型转换

源/目的数据类型支持组合如下表所示，未列出的组合不支持。表中“支持的有效舍入模式”表示能够实际改变舍入结果的模式；CAST_NONE可用于所有已列出的转换，并按上一节所述规则处理。接口缺省传入CAST_ROUND。

| 源数据类型 | 目的数据类型 | 支持的有效舍入模式 | 特殊说明 |
|---|---|---|---|
| DT_FP32 | DT_FP32 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | 同类型舍入，不是原值拷贝 |
| DT_FP32 | DT_FP16 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC、CAST_ODD | - |
| DT_FP32 | DT_BF16 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | - |
| DT_FP32 | DT_INT16、DT_INT32、DT_INT64 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | CAST_NONE使用CAST_TRUNC |
| DT_FP32 | DT_FP8E4M3FN、DT_FP8E5M2 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | - |
| DT_FP32 | DT_HF8 | 固定CAST_ROUND | 其他mode不改变舍入行为 |
| DT_FP16 | DT_FP32 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_FP16 | DT_INT8、DT_UINT8、DT_INT16、DT_INT32 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | CAST_NONE使用CAST_TRUNC |
| DT_FP16 | DT_HF8 | 固定CAST_ROUND | 其他mode不改变舍入行为 |
| DT_BF16 | DT_FP32 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_BF16 | DT_INT32 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | CAST_NONE使用CAST_TRUNC |
| DT_BF16 | DT_FP16 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | - |
| DT_BF16 | DT_FP4E2M1、DT_FP4E1M2 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | FP4为打包类型，两个元素共用一个字节 |
| DT_FP4E2M1、DT_FP4E1M2 | DT_BF16 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_UINT8 | DT_FP16、DT_UINT16 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_INT8 | DT_FP16、DT_INT16、DT_INT32 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_INT16 | DT_FP16 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | - |
| DT_INT16 | DT_FP32、DT_UINT32、DT_INT32 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_INT16 | DT_UINT8 | 不涉及舍入 | 整数窄化，mode不影响结果 |
| DT_INT32 | DT_FP32 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | - |
| DT_INT32 | DT_INT64 | 不涉及舍入 | 类型扩展，mode不影响结果 |
| DT_INT32 | DT_UINT8、DT_INT16、DT_UINT16 | 不涉及舍入 | 整数窄化，mode不影响结果 |
| DT_UINT32 | DT_UINT8、DT_UINT16、DT_INT16 | 不涉及舍入 | 整数窄化，mode不影响结果 |
| DT_INT64 | DT_FP32 | CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC | - |
| DT_INT64 | DT_INT32 | 不涉及舍入 | 整数窄化，mode不影响结果 |
| DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8 | DT_FP32 | 不涉及舍入 | 类型扩展，mode不影响结果 |

### 其他约束

- out和src必须均为UB Tile。转换范围由out.valid_shape确定，src的有效区域必须覆盖该范围；源、目的Tile的物理shape和行跨度可以不同，但必须保证转换区域内的元素能够按相同逻辑坐标逐元素对应。接口只定义out有效区域内的转换结果，不应依赖该区域之外的内容。
- DT_FP32 -> DT_FP16支持out和src使用完全相同的UB起始地址进行原地转换。除该场景外，不保证源、目的存储区域发生部分重叠或完全重叠时的转换结果；使用其他数据类型组合时，应为out和src分配互不重叠的UB存储区域。
- 对DT_FP4E2M1和DT_FP4E1M2，Tile的shape按逻辑元素计数，两个相邻元素打包在一个字节中，因此物理shape的最后一维必须为2的倍数。执行DT_BF16到FP4的转换时，valid_shape的最后一维也应为偶数，以保证每个有效元素均能组成完整的打包字节。
- mode必须在编译期确定，不能使用运行期Scalar或Tensor值动态选择。

## 返回值说明

无。

## 调用示例

### FP16→FP32扩展

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def cast_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    src_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    out_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    src_tile = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=out_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        src_current = src_tile.current()
        out_current = out_tile.current()
        pl.load(src_current, src, [0, 0])
        pl.cast(out_current, src_current, mode=pl.RoundMode.CAST_NONE)
        pl.store(out, out_current, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:cast:start -->
```bash
输入数据src：[[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
输出数据out：[[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
```
<!-- pypto-doc-output:cast:end -->

### 舍入模式

```python
# FP32 -> FP16：舍入到最近值，中间值取偶数。
pl.cast(dst_fp16, src_fp32, mode=pl.RoundMode.CAST_RINT)

# FP32 -> INT32：向负无穷方向舍入。
# 例如，1.6转换为1，-1.6转换为-2。
pl.cast(dst_int32, src_fp32, mode=pl.RoundMode.CAST_FLOOR)

# FP16 -> FP32：类型扩展，不需要舍入。
pl.cast(dst_fp32, src_fp16, mode=pl.RoundMode.CAST_NONE)
```
