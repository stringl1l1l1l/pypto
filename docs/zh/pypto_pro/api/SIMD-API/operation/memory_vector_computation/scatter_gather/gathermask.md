# pypto_pro.language.gathermask

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

按位模式从源Tile中抽取列到目标Tile，抽取的列由pattern_mode参数决定。

## 函数原型

```python
pypto_pro.language.gathermask(out: Tile, src: Tile, *, pattern_mode: int) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标Tile，存放按位模式抽取的列。dtype与src一致，行数与src一致，列数由pattern_mode决定。有效列必须连续存储。 |
| src | 输入 | 源Tile，位于UB的行主序Tile，dtype支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64，out与src元素位宽相同。 |
| pattern_mode | 输入 | 位模式，编译期常量整数，取值1～7，决定抽取哪些列：<br>1：取偶数列，即`src[:, 0::2]`。<br>2：取奇数列，即`src[:, 1::2]`。<br>3：每4列取第1列，即`src[:, 0::4]`。<br>4：每4列取第2列，即`src[:, 1::4]`。<br>5：每4列取第3列，即`src[:, 2::4]`。<br>6：每4列取第4列，即`src[:, 3::4]`。<br>7：取全部列，等价于copy。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p1_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=1)
        pl.store(dst, tile_dst, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:gathermask:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据pattern_mode：1
输出数据dst：[[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
```
<!-- pypto-doc-output:gathermask:end -->

### 其他典型用法

```python
# 抽取奇数列
pl.gathermask(tile_dst, tile_src, pattern_mode=2)

# 全取（copy）
pl.gathermask(tile_dst, tile_src, pattern_mode=7)
```
