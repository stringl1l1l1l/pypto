# pypto.experimental.online\_softmax

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

该接口为定制接口，约束较多。不保证稳定性。

该算子用于块式在线Softmax计算，对输入scores按第0维做局部统计。算子会先对scores乘以scale，再计算每一列的最大值和指数和，同时输出未归一化的指数结果。该接口通常用于FlashAttention等分块注意力场景，配合`pypto.experimental.online_softmax_update`逐块更新全局最大值、指数和与中间输出。

## 函数原型

```python
online_softmax(scores: Tensor, scale: float) -> Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| scores | 输入 | 源操作数。<br>支持的数据类型为：DT_FP32。<br>不支持空Tensor，支持两维。<br>形状为[k_len, q_len]。 |
| scale | 输入 | float类型。<br>对scores进行缩放的标量，常用取值为`1.0 / sqrt(head_dim)`。 |

## 返回值说明

返回三个输出Tensor：

| 返回值 | 说明 |
|--------|------|
| exp_scores_bf16 | 缩放并减去列最大值后的指数结果，数据类型为DT_BF16，Shape与scores相同。 |
| column_max | 每一列的局部最大值，数据类型为DT_FP32，Shape为[1, q_len]。 |
| column_sum | 每一列的局部指数和，数据类型为DT_FP32，Shape为[1, q_len]。 |

## 约束说明

1. 该接口为定制接口，不保证稳定性。
2. scores 数据类型仅支持 DT_FP32。
3. 当前版本不切分第0维，要求 scores.shape[0] <= vec_tile[0]。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape的维度设置须与输入scores保持一致。当前版本不切分第0维，要求`scores.shape[0] <= vec_tile[0]`。

### 接口调用示例

```python
import pypto

scores = pypto.tensor([128, 128], pypto.DT_FP32)
scale = 1.0 / (128 ** 0.5)

pypto.set_vec_tile_shapes(128, 64)
exp_scores_bf16, column_max, column_sum = pypto.experimental.online_softmax(scores, scale)
```
