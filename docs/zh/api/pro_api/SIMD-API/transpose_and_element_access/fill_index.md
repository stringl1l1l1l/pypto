# pypto_pro.language.fill_index

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

以start为起始值，产生连续整数序列填入目标Tile：out[j] = start + j。

典型场景：生成位置编码所需的递增索引，或初始化用于排序、gather的索引。

## 函数原型

```python
pypto_pro.language.fill_index(
    out: Tile,
    start: Union[int, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标Tile，存放生成的索引序列。shape为[1, N]（行数为1，列数为索引个数），须位于UB。 |
| start | 输入 | 起始值，支持整数或运行时整型标量表达式，生成代码时转换为out的元素类型。 |

## 约束说明

- out支持DT_INT16、DT_UINT16、DT_INT32和DT_UINT32。
- out必须为单行Tile（shape第0维为1）。
- out须位于UB。
- 生成的索引序列为[start, start+1, ..., start+N-1]，其中N为out的有效列数（valid_shape[1]）。
- 必须在section_vector内调用。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl

START = 0


@pl.jit(auto_mutex=True)
def fill_index_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tt = pl.TileType(shape=[1, 128], valid_shape=[-1, -1],
                     dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        m_dim = out.shape[0]
        n_dim = out.shape[1]
        for i in pl.range(0, m_dim, 1):
            cur_out = tile_out.current()
            pl.set_validshape(cur_out, [1, n_dim])
            pl.fill_index(cur_out, START)
            pl.store(out, cur_out, [i, 0])
```
