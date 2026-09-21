# pypto.set_convbp_input_tile_shapes

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

设置卷积反向（conv backward input）计算中L1/L0缓存层级下各维度的TileShape（切片形状）大小。

卷积反向Dx算子可视为矩阵乘法 `(HinWin, CoutKhKw) × (CoutKhKw, Cin)`，其中M/N/K三轴分别对应 `HinWin` / `Cin` / `Cout×Kh×Kw`。

## 函数原型

```python
set_convbp_input_tile_shapes(l1_info: pypto_impl.ConvBpTileL1Info, l0_info: pypto_impl.ConvBpTileL0Info) -> None
```

## 参数说明

| 参数名   | 输入/输出 | 说明                                         |
|----------|-----------|----------------------------------------------|
| l1_info  | 输入      | L1缓存层级下卷积反向计算的TileShape配置信息，有tileML1、tileNL1、tileKL1三个参数。请参考[约束说明](#约束说明)章节填写 |
| l0_info  | 输入      | L0缓存层级下卷积反向计算的TileShape配置信息，有tileML0、tileNL0、tileKL0三个参数。请参考[约束说明](#约束说明)章节填写 |

## 返回值说明

void

## 约束说明

TileShape需要满足以下约束条件：

- 对齐约束：

    - ConvBpTileL1Info各维度值需满足范围约束：

        - tileML1：需小于Win或者为Win的整数倍，即 `tileML1 <= Win` 或 `tileML1 % Win == 0`（HinWin合轴切分，M方向保证不跨Win行）

        - tileNL1：需为Cin0的倍数，即 `tileNL1 % 16 == 0`

        - tileKL1：需为 `Cout0 × Kh × Kw` 的倍数，即 `tileKL1 % (16 * Kh * Kw) == 0`

    - ConvBpTileL0Info各维度值需满足对齐约束：

        - tileML0：需为16的倍数，即 `tileML0 % 16 == 0`，且 `tileML0 <= CeilAlign(tileML1, 16)`

        - tileNL0：需为16的倍数，即 `tileNL0 % 16 == 0`，且 `tileNL0 <= tileNL1`

        - tileKL0：需为16的倍数，即 `tileKL0 % 16 == 0`，且 `tileKL0 <= tileKL1`

    其中：

    - `Cin0 = 16`
    - `Cout0 = 16`

- buffer空间约束：

    - L0A、L0B、L0C空间约束：

        ```txt
        tileML0 * tileKL0 * sizeof(dtype) <= L0A_size

        tileKL0 * tileNL0 * sizeof(dtype) <= L0B_size

        tileML0 * tileNL0 * sizeof(FP32) <= L0C_size
        ```

        其中：

        - `L0A_size = 65536 bytes`
        - `L0B_size = 65536 bytes`
        - `L0C_size = 131072 bytes`

    - L1空间约束：

        ```txt
        (L1Weight + houtL1 * woutL1 * coutL1) * sizeof(dtype) <= L1_size
        ```

        其中：

        - `L1Weight = tileKL1 * tileNL1`
        - `houtL1 = min(CeilDiv(tileML1, Win) + (Kh - 1) * dilationH, (Hout - 1) * strideH + 1)`
        - `woutL1 = min(min(tileML1, Win) + (Kw - 1) * dilationW, (Wout - 1) * strideW + 1)`
        - `coutL1 = tileKL1 / (Kh * Kw)`
        - `L1_size = 524288 bytes`（512KB）
        - `dtype为输入矩阵的数据类型`
        - `CeilDiv(a, b) = (a + b - 1) // b`

## 调用示例

```python
# 构造L1 Tile配置（确保各值在合法范围）
l1_tile_info = pypto.pypto_impl.ConvBpTileL1Info(
    tileML1=16,    # HinWin切分，需满足 <= Win 或 % Win == 0
    tileNL1=16,    # Cin切分，需为16的倍数
    tileKL1=144    # Cout*Kh*Kw切分，需为16*Kh*Kw的倍数（如Kh=3,Kw=3时16*9=144）
)

# 构造L0 Tile配置（满足对齐约束）
l0_tile_info = pypto.pypto_impl.ConvBpTileL0Info(
    tileML0=16,    # 需为16的倍数，且不大于CeilAlign(tileML1, 16)
    tileNL0=16,    # 需为16的倍数，且不大于tileNL1
    tileKL0=16     # 需为16的倍数，且不大于tileKL1
)

# 设置卷积反向TileShape
pypto.set_convbp_input_tile_shapes(l1_tile_info, l0_tile_info)
```
