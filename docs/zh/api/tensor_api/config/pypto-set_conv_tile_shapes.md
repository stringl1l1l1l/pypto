# pypto.set_conv_tile_shapes

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

设置卷积（conv）计算中L1/L0缓存层级下各维度的TileShape（切片形状）大小，同时控制L0TileInfo配置开关的使能状态。

## 函数原型

```python
set_conv_tile_shapes(tile_l1_info: pypto_impl.TileL1Info, tile_l0_info: pypto_impl.TileL0Info = None) -> None
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                   |
|--------------|-----------|--------------------------------------|
| tile_l1_info | 输入      | L1缓存层级下卷积计算的TileShape配置信息 |
| tile_l0_info | 输入      | L0缓存层级下卷积计算的TileShape配置信息 |

## 返回值说明

void

## 约束说明

TileShape需要满足以下约束条件：

**注意：L1和L0上都有tileN，但是其代表的含义不同，tileL1Info.tileN代表输出通道数，tileL0Info.tileN代表L0层级的n的大小**

- 对齐约束：

    - tileL1Info各维度值需满足范围约束：

        - 1 <= tileHin <= Hin（Hin为输入特征图实际高度）

        - wout % 16 = 0时， 1 <= tileHout <= Hout（Hout为输出特征图实际高度）

        - wout % 16 != 0时， tileHout = 1

        - 1 <= tileWin <= Win（Win为输入特征图实际宽度）

        - 1 <= tileWout <= CeilAlign(Wout, 16)（Wout为输出特征图实际宽度），tileWout需满足16元素对齐，即`tileWout % 16 == 0`

        - C0 <= tileCinFmap <= CeilAlign(Cin, C0)（Cin为输入特征图实际通道数）

        - tileCinFmap * sizeof(dtype) % 32 == 0

        - C0 <= tileCinWeight <= CeilAlign(Cin, C0)（Cin为权重输入通道实际数量）

        - tileCinWeight * sizeof(dtype) % 32 == 0

        - 1 <= tileN <= CeilAlign(Cout // groups, 16)（Cout为输出特征图实际通道数）

        - tileN % 16 == 0

        - tileBatch = 1（代表batch数）

    - tileL0Info各维度值需满足对齐约束：

        - tileK需满足：`C0 <= tileK <= min(kAL1, kBL1)`

        - tileK需满足：`tileK % C0 == 0`

        - tileK需满足： `kAL1 % tilek == 0`

        - tilek需满足： `kBL1 % tilek == 0`

        - tileW需满足16元素对齐，即`tileW % 16 == 0`

        - tileW需满足： `1 <= tileW <= tileWout`

        - tileH需满足： `1 <= tileH <= tileHout`

        - tileN（代表L0上的n的大小）需满足16元素对齐，即`tileN % 16 == 0`

        - tileN需满足： `1 <= tileN <= CeilAlign(tileL1Info.tileN, 16)`

        其中：

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

    - L0与L1维度层级约束：

        - 1 <= tileL0Info.tileH <= tileL1Info.tileHout且tileL1Info.tileHout % tileL0Info.tileH == 0

        - 1 <= tileL0Info.tileW <= tileL1Info.tileWout且tileL1Info.tileWout % tileL0Info.tileW == 0

        - 1 <= tileL0Info.tileN <= tileL1Info.tileN

- buffer空间约束：

    - L0A、L0B、L0C空间约束：

        ```txt
        CeilAlign(tileH * tileW, 16)* CeilAlign(tileK, C0) * sizeof(dtype) <= L0A_size

        CeilAlign(tileK, C0) * CeilAlign(tileN, 16) * sizeof(dtype) <= L0B_size

        CeilAlign(tileH * tileW, 16)* CeilAlign(tileN, 16) * sizeof(FP32) <= L0C_size
        ```

        其中：

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `L0A_size = 65536 Byte`

        - `L0B_size = 65536 Byte`

        - `L0C_size = 131072 Byte`

        - `ALIGN_SIZE_32 = 32`

    - L1空间约束：

        ```txt
        CeilAlign(hinL1 * winL1 * tileCinFmap * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(nL1 * kBL1 * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(tileN * sizeof(dtype), ALIGN_SIZE_32) <= L1_size
        ```

        其中：

        - `hinL1 = min((tileHout - 1) * strideH + (Kh - 1) * dilationH + 1, Hin)` （Hin为输入特征图高度）

        - `winL1 = min((tileWout - 1) * strideW + (Kw - 1) * dilationW + 1, Win)` （Win为输入特征图宽度）

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `nL1 = tileN`（输出通道数）

        - `dtype为input_conv（输入矩阵）的数据类型`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

        - `CeilAlign(value, align) {  return ((value + align - 1) // align) * align;}`

- 特殊场景约束：

    - 当未传入`tileL0Info`时，自动使用默认`TileL0Info`实例，且L0TileInfo开关自动关闭；传入有效`tileL0Info`时，L0TileInfo开关自动开启。

    - 卷积核/通道维度配置需与实际卷积算子的输入输出通道数、核大小匹配，避免Tile大小超出算子维度范围。

## 调用示例

```python
# 构造L1 Tile配置（确保各值在合法范围）
l1_tile = pypto_impl.TileL1Info(
    tileHin=4,        # 需满足1 <= tileHin <= Hin
    tileHout=4,       # 需满足1 <= tileHout <= Hout
    tileWin=8,        # 需满足1 <= tileWin <= Win
    tileWout=8,       # 需满足1 <= tileWout <= Wout
    tileCinFmap=16,   # 需满足1 <= tileCinFmap <= Cin
    tileCinWeight=32, # 需满足1 <= tileCinWeight <= Cin
    tileN=16,         # 需满足1 <= tileN <= Cout
    tileBatch=1       # 需满足tileBatch = 1
)

# 构造L0 Tile配置（满足对齐约束）
l0_tile = pypto_impl.TileL0Info(
    tileH=2,   # 需满足tileH <= tileL1Info.tileHout且tileL1Info.tileHout % tileH == 0
    tileW=8,   # 需满足tileW <= tileL1Info.tileWout且tileL1Info.tileWout % tileW == 0
    tileK=32,  # 需满足tileK % C0 == 0（C0为对齐基数，由数据类型决定）
    tileN=16   # 需满足tileN % 16 == 0
)

# 设置卷积TileShape（开启L0TileInfo）
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile, tile_l0_info=l0_tile)

# 仅设置L1 TileShape（关闭L0TileInfo）
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile)
```
