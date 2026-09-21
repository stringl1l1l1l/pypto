# pypto.cast

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

根据源操作数和目的操作数Tensor的数据类型进行精度转换，如果目的操作数是整型且源操作数的数值超过整型的数据表示范围，进行精度转换结果为目的操作数的最大值或者最小值。

## 注意事项

- **PyPTO Tensor不支持`.to()`方法**：PyPTO Tensor没有`.to(dtype)`方法，必须使用`pypto.cast(tensor, dtype)`进行数据类型转换

- **舍入模式详细说明**：浮点数表示方式、二进制舍入规则及各CastMode的具体行为，请参考[CastMode](../datatype/CastMode.md)。

## 函数原型

```python
cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE,
     satmode: SaturationMode = SaturationMode.OFF) -> Tensor
```

## 参数说明

| 参数名     | 输入/输出 | 说明                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| input      | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_INT32，DT_INT64，DT_INT4，DT_FP8E4M3，DT_FP8E5M2，DT_HF8，DT_FP4_E2M1，DT_FP4_E1M2。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dtype      | 输入      | 精度转换后的数据类型。<br>支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_INT32，DT_INT64，DT_INT4，DT_FP8E4M3，DT_FP8E5M2，DT_HF8，DT_FP4_E2M1，DT_FP4_E1M2。 |
| CastMode   | 输入      | 源操作数枚举类型，用以控制精度转换处理模式，具体定义为：[CastMode](../datatype/CastMode.md)。<br>默认为CAST_NONE，常见类型之间的转换，框架会自动转换，与torch对齐，详见约束说明。 |
| SaturationMode    | 输入      | 饱和模式枚举类型，用以控制浮点数转整数时的溢出处理方式，具体定义为：[SaturationMode](../datatype/SaturationMode.md)。<br>默认为OFF（截断模式），当设置为ON时，超出目标类型范围的数值会被截断到最大值或最小值（饱和截断），详见约束说明。 |

## 返回值说明

| 类型 | 说明 |
|:-----|:-----|
| Tensor | 数据类型为dtype的新Tensor，其值与input相同。 |

## 约束说明

> **CastMode舍入模式详细说明请参考**：[CastMode](../datatype/CastMode.md)

<!-- npu="A3,910b" id4 -->

### Atlas A3系列产品/Atlas A2系列产品支持的转换

| 源类型 | 目标类型 | 支持的CastMode | 默认CastMode | 特殊说明 |
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP16 | RINT，ROUND，FLOOR，CEIL，TRUNC，ODD | CAST_RINT | - |
| DT_FP32 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | 同类型舍入 |
| DT_FP32 | DT_BF16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_FP32 | DT_INT64 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 支持inf/-inf等边缘情况 |
| DT_FP16 | DT_FP32 | 不支持舍入模式 | - | 类型扩展 |
| DT_FP16 | DT_INT32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_FP16 | DT_INT16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 支持inf/-inf等边缘情况 |
| DT_FP16 | DT_INT8 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 支持inf/-inf等边缘情况 |
| DT_FP16 | DT_UINT8 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_FP16 | DT_INT4 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 打包类型，每字节包含2个元素 |
| DT_BF16 | DT_FP32 | 不支持舍入模式 | - | 类型扩展 |
| DT_BF16 | DT_INT32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_INT32 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_INT32 | DT_INT64 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT32 | DT_INT16 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_INT32 | DT_FP16 | 不支持舍入模式 | - | deq模式，需设置deqscale |
| DT_INT16 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_INT16 | DT_FP16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_INT64 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_INT64 | DT_INT32 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_UINT8 | DT_FP16 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT8 | DT_FP16 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT4 | DT_FP16 | 不支持舍入模式 | - | 打包类型，每字节包含2个元素 |
<!-- end id4 -->

<!-- npu="950" id5 -->

### Ascend 950PR&950DT系列产品支持的转换

Ascend 950PR&950DT系列产品使用不同的CastMode体系，内部实现基于 `RoundRType`/`RoundAType`/`RoundFType`/`RoundCType`/`RoundZType`/`RoundOType` 等模板参数，用户接口层面仍使用统一的CastMode enum。

| 源类型 | 目标类型 | 支持的CastMode | 默认CastMode | 特殊说明 |
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | 同类型舍入（vtrc指令） |
| DT_FP32 | DT_FP16 | RINT，ROUND，FLOOR，CEIL，TRUNC，ODD | CAST_RINT | - |
| DT_FP32 | DT_BF16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_FP32 | DT_INT64 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 支持inf/-inf等边缘情况 |
| DT_FP32 | DT_FP8E4M3 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_FP32 | DT_FP8E5M2 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_FP32 | DT_HF8 | **仅支持ROUND** | CAST_ROUND | H8必须使用ROUND_A，不支持其他模式 |
| DT_FP16 | DT_FP32 | 不支持舍入模式 | - | 类型扩展（PART_EVEN） |
| DT_FP16 | DT_INT32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | ROUND_PART模式 |
| DT_FP16 | DT_INT16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 支持inf/-inf等边缘情况，ROUND_SAT模式 |
| DT_FP16 | DT_INT8 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | 支持inf/-inf等边缘情况，ROUND_SAT_PART模式 |
| DT_FP16 | DT_UINT8 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | ROUND_SAT_PART模式 |
| DT_FP16 | DT_HF8 | **仅支持ROUND** | CAST_ROUND | H8必须使用ROUND_A，不支持其他模式 |
| DT_BF16 | DT_FP32 | 不支持舍入模式 | - | 类型扩展（PART_EVEN） |
| DT_BF16 | DT_INT32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_TRUNC | ROUND_SAT_PART模式 |
| DT_BF16 | DT_FP16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | SAT_ROUND模式（先饱和后舍入） |
| DT_UINT8 | DT_FP16 | 不支持舍入模式 | - | 类型扩展 |
| DT_UINT8 | DT_UINT16 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT8 | DT_FP16 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT8 | DT_INT16 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT8 | DT_INT32 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT16 | DT_UINT8 | 不支持舍入模式 | - | SAT_PART模式 |
| DT_INT16 | DT_FP16 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | ROUND模式 |
| DT_INT16 | DT_FP32 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT16 | DT_UINT32 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT16 | DT_INT32 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT32 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | ROUND模式 |
| DT_INT32 | DT_INT16 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_INT32 | DT_UINT16 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_INT32 | DT_INT64 | 不支持舍入模式 | - | 类型扩展 |
| DT_INT32 | DT_UINT8 | 不支持舍入模式 | - | SAT_PART模式 |
| DT_INT32 | DT_FP16 | 不支持舍入模式 | - | deq模式，需设置deqscale |
| DT_UINT32 | DT_UINT8 | 不支持舍入模式 | - | SAT_PART模式 |
| DT_UINT32 | DT_UINT16 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_UINT32 | DT_INT16 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_INT64 | DT_FP32 | RINT，ROUND，FLOOR，CEIL，TRUNC | CAST_RINT | - |
| DT_INT64 | DT_INT32 | 不支持舍入模式 | - | 仅饱和控制 |
| DT_FP8E4M3 | DT_FP32 | 不支持舍入模式 | - | 类型扩展 |
| DT_FP8E5M2 | DT_FP32 | 不支持舍入模式 | - | 类型扩展 |
| DT_HF8 | DT_FP32 | 不支持舍入模式 | - | 类型扩展 |
| DT_BF16 | DT_FP4_E2M1 | 不支持舍入模式 | - | - |
| DT_BF16 | DT_FP4_E1M2 | 不支持舍入模式 | - | - |
| DT_FP4_E2M1 | DT_BF16 | 不支持舍入模式 | - | - |
| DT_FP4_E1M2 | DT_BF16 | 不支持舍入模式 | - | - |
<!-- end id5 -->

### 饱和模式设置说明

饱和模式（SaturationMode）用于控制浮点数转整数时的溢出处理方式：

- **OFF（默认）**：截断模式，超出目标类型范围的数值按二进制截断
- **ON**：饱和模式，超出目标类型范围的数值被截断到最大值或最小值

对于量化等场景，建议设置`satmode=SaturationMode.ON`，避免溢出导致的精度问题。其他场景使用默认值即可。

---

### 其他约束

1. 当cast前后类型相同的时候，某些场景下会产生空操作，不保证精度。

2. **DT_INT4（S4）特殊说明**：打包类型，每字节包含2个元素，仅支持与DT_FP16的相互转换。

3. **DT_FP4_E2M1/DT_FP4_E1M2特殊说明**：
    <!-- npu="950" id6 -->
    - Ascend 950PR&950DT系列产品：支持
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - Atlas A3系列产品：不支持
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - Atlas A2系列产品：不支持
    <!-- end id8 -->
    - 使用逻辑4-bit类型，与scaled_mm一致；输入参数标注和cast目标类型均使用不带X2的枚举。
    - 转换前后逻辑Shape保持不变；每字节存储2个FP4元素，逻辑末轴长度必须为偶数。
    - 与Torch交互时，用torch.uint8承载打包数据，存储Shape为[m, n/2]；参数标注为`pypto.Tensor([], pypto.DT_FP4_E1M2)`或`pypto.Tensor([], pypto.DT_FP4_E2M1)`后，kernel内看到的逻辑Shape为[m, n]。
    - 仅支持与DT_BF16的相互转换

4. **DT_HF8 (hifloat8)特殊说明**：
    <!-- npu="950" id9 -->
    - Ascend 950PR&950DT系列产品：支持
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - Atlas A3系列产品：不支持
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - Atlas A2系列产品：不支持
    <!-- end id11 -->
    - 必须使用CAST_ROUND舍入模式（对应硬件的ROUND_A）
    - 如果指定其他CastMode，会自动回退到CAST_ROUND

5. **不支持的CastMode处理**：
    - 当用户为某个转换指定了硬件不支持的CastMode时，框架不会报错
    - 框架会自动采用该转换的默认CastMode：
      - 浮点→整数：采用CAST_TRUNC
      - 其他场景：采用CAST_RINT

6. **deq模式说明**：INT32→FP16转换使用deq模式，需要通过`set_deqscale`设置缩放因子，默认值为1.0。
7. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2], pypto.DT_FP32)
y = pypto.cast(x, pypto.DT_FP16)
```

结果示例如下：

```python
输入数据x: [2.0, 3.0] # x.dtype: pypto.DT_FP32

输出数据y: [2.0, 3.0] # y.dtype: pypto.DT_FP16
```

#### 使用饱和模式（推荐用于浮点数转整数）

```python
# 示例1：FP16转INT8，使用饱和模式防止溢出
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
# 输出：[127, -128, 50]

# 示例2：FP16转INT8，使用饱和模式防止溢出
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.OFF)
# 输出：[44, -44, 50]
```
