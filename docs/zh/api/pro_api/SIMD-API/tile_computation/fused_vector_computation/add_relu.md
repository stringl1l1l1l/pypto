# pypto_pro.language.add_relu

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

对两个Tile执行逐元素加法，再对结果执行ReLU激活，将小于0的元素置为0。与add_relu_cast相比，本接口不进行数据类型转换。

## 函数原型

```python
pypto_pro.language.add_relu(
  out: Tile,
  lhs: Tile,
  rhs: Tile
) -> None:
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标 UB Tile，用于保存逐元素加法和 ReLU 激活的结果。支持的数据类型为 DT_FP16、DT_FP32 和 DT_INT32。 |
| lhs | 输入/输出 | 左操作数 UB Tile，在计算过程中用于保存加法的中间结果。支持的数据类型为 DT_FP16、DT_FP32 和 DT_INT32。 |
| rhs | 输入 | 右操作数 UB Tile。支持的数据类型为 DT_FP16、DT_FP32 和 DT_INT32。 |

## 约束说明

- out、lhs 和 rhs 的数据类型和有效 Shape 须一致。
- out、lhs 和 rhs 支持 ND、ZN 和 ZZ。
- lhs 在计算过程中会被修改，调用后其中的原始数据不再保留。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                    out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(
      shape=[64, 64],
      dtype=pl.DT_FP32,
      target_memory=pl.MemorySpace.Vec,
      layout=pl.TensorLayout.ND,
    )
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:add_relu:start -->
```bash
输入数据a：[[-6 -5.75 -5.5 -5.25 -5 -4.75 -4.5 -4.25 ...], [10 10.25 10.5 10.75 11 11.25 11.5 11.75 ...], [26 26.25 26.5 26.75 27 27.25 27.5 27.75 ...], [42 42.25 42.5 42.75 43 43.25 43.5 43.75 ...], ...]
输入数据b：[[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
输出数据out：[[0 0 0 0 0 0 0 0.25 ...], [43 43.75 44.5 45.25 46 46.75 47.5 48.25 ...], [91 91.75 92.5 93.25 94 94.75 95.5 96.25 ...], [139 139.75 140.5 141.25 142 142.75 143.5 144.25 ...], ...]
```
<!-- pypto-doc-output:add_relu:end -->
