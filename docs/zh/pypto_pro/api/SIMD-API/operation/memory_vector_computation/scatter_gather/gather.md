# pypto_pro.language.gather

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

根据索引Tile中的扁平元素偏移，从源Tile中聚合元素到目标Tile，即`dst_flat[i] = src_flat[idx[i]]`。与[pypto_pro.language.scatter](scatter.md)互为反向操作。

## 函数原型

```python
pypto_pro.language.gather(
    out: Tile,
    src: Tile,
    idx: Tile,
    tmp: Tile,
    *,
    cmp_mode: int = 0,
    offset: int = 0,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标Tile，存放按索引聚合的结果。dtype与src一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8。shape与valid_shape须与索引结果匹配。 |
| src | 输入 | 源Tile，位于UB的行主序Tile，dtype与out一致。 |
| idx | 输入 | 索引Tile，位于UB，支持DT_INT16、DT_UINT16、DT_INT32、DT_UINT32，64 bit数据须使用32 bit索引。元素值为src中的扁平元素索引，越界行为未定义。 |
| tmp | 输入 | 临时工作Tile，供硬件中间计算使用。位于UB，dtype与idx一致，shape与idx一致，不可与out、src、idx重叠。 |
| cmp_mode | 输入 | 可选，比较模式，取0表示不比较。 |
| offset | 输入 | 可选，对idx中的索引值施加的偏移。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gather_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    indices: pl.Tensor[[64, 128], pl.DT_INT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt_fp16 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_int32 = pl.TileType(shape=[64, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_fp16, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_int32, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt_int32, addrs=0xC000, mutex_ids=[2])
    tile_dst = pl.make_tile_group(type=tt_fp16, addrs=0x14000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_tmp = tile_tmp.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, indices, [0, 0])
        pl.gather(cur_dst, cur_src, cur_idx, cur_tmp)
        pl.store(dst, cur_dst, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:gather:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 67 67.25 67.5 67.75 68 68.25 68.5 68.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 99 99.25 99.5 99.75 100 100.25 100.5 100.75 ...], ...]
输入数据indices：[[127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 ...], [255 254 253 252 251 250 249 248 247 246 245 244 243 242 241 240 ...], [383 382 381 380 379 378 377 376 375 374 373 372 371 370 369 368 ...], [511 510 509 508 507 506 505 504 503 502 501 500 499 498 497 496 ...], ...]
输出数据dst：[[32.75 32.5 32.25 32 31.75 31.5 31.25 31 30.75 30.5 30.25 30 29.75 29.5 29.25 29 ...], [64.75 64.5 64.25 64 63.75 63.5 63.25 63 62.75 62.5 62.25 62 61.75 61.5 61.25 61 ...], [96.75 96.5 96.25 96 95.75 95.5 95.25 95 94.75 94.5 94.25 94 93.75 93.5 93.25 93 ...], [128.75 128.5 128.25 128 127.75 127.5 127.25 127 126.75 126.5 126.25 126 125.75 125.5 125.25 125 ...], ...]
```
<!-- pypto-doc-output:gather:end -->

### 带偏移的用法

```python
# offset会在gather时参与索引计算
pl.gather(cur_dst, cur_src, cur_idx, cur_tmp, offset=16)
```
