# pypto_pro.language.scatter

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

根据索引Tile中的扁平元素偏移，将源Tile的元素分散写入目标Tile的对应位置，即`dst_flat[indices[i,j]] = src[i,j]`。

## 函数原型

```python
pypto_pro.language.scatter(out: Tile, src: Tile, idx: Tile) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标Tile，按索引分散写入。位于UB，dtype与src一致，支持的数据类型详见[约束说明](#约束说明)。 |
| src | 输入 | 源Tile，dtype与out一致，shape与out一致。 |
| idx | 输入 | 索引Tile，位于UB，为有符号或无符号整数Tile，数据类型与索引类型的组合详见[约束说明](#约束说明)。有效shape须与被分散的源数据匹配，索引越界或发生写冲突时行为未定义。 |

## 约束说明

- 硬件TSCATTER指令仅支持以下数据类型与索引类型的组合：

  | 数据类型 | 索引类型 | 说明 |
  |---|---|---|
  | DT_INT64、DT_UINT64 | DT_INT32、DT_UINT32 | 64 bit数据使用32 bit索引。 |
  | DT_FP32、DT_INT32、DT_UINT32 | DT_INT32、DT_UINT32 | 32 bit数据使用32 bit索引。 |
  | DT_FP16、DT_BF16、DT_INT16、DT_UINT16 | DT_INT16、DT_UINT16 | 16 bit数据使用16 bit索引。 |
  | DT_INT8、DT_UINT8 | DT_INT16、DT_UINT16 | 8 bit数据使用16 bit索引。 |


## 返回值说明

无。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def scatter_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    indices: pl.Tensor[[64, 128], pl.DT_INT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x8000, mutex_ids=[1])
    tile_dst = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, indices, [0, 0])
        pl.scatter(cur_dst, cur_src, cur_idx)
        pl.store(dst, cur_dst, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:scatter:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 67 67.25 67.5 67.75 68 68.25 68.5 68.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 99 99.25 99.5 99.75 100 100.25 100.5 100.75 ...], ...]
输入数据indices：[[8191 8190 8189 8188 8187 8186 8185 8184 8183 8182 8181 8180 8179 8178 8177 8176 ...], [8063 8062 8061 8060 8059 8058 8057 8056 8055 8054 8053 8052 8051 8050 8049 8048 ...], [7935 7934 7933 7932 7931 7930 7929 7928 7927 7926 7925 7924 7923 7922 7921 7920 ...], [7807 7806 7805 7804 7803 7802 7801 7800 7799 7798 7797 7796 7795 7794 7793 7792 ...], ...]
输出数据dst：[[2.048750e+03 2.048500e+03 2.048250e+03 2.048000e+03 2.047750e+03 2.047500e+03 2.047250e+03 2.047000e+03 2.046750e+03 2.046500e+03 2.046250e+03 2.046000e+03 2.045750e+03 2.045500e+03 2.045250e+03 2.045000e+03 ...], [2.016750e+03 2.016500e+03 2.016250e+03 2.016000e+03 2.015750e+03 2.015500e+03 2.015250e+03 2.015000e+03 2.014750e+03 2.014500e+03 2.014250e+03 2.014000e+03 2.013750e+03 2.013500e+03 2.013250e+03 2.013000e+03 ...], [1.984750e+03 1.984500e+03 1.984250e+03 1.984000e+03 1.983750e+03 1.983500e+03 1.983250e+03 1.983000e+03 1.982750e+03 1.982500e+03 1.982250e+03 1.982000e+03 1.981750e+03 1.981500e+03 1.981250e+03 1.981000e+03 ...], [1.952750e+03 1.952500e+03 1.952250e+03 1.952000e+03 1.951750e+03 1.951500e+03 1.951250e+03 1.951000e+03 1.950750e+03 1.950500e+03 1.950250e+03 1.950000e+03 1.949750e+03 1.949500e+03 1.949250e+03 1.949000e+03 ...], ...]
```
<!-- pypto-doc-output:scatter:end -->
