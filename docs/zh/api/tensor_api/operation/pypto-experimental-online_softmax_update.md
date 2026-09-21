# pypto.experimental.online\_softmax\_update

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

该算子用于在线Softmax的状态更新。给定历史块的最大值、指数和、中间输出，以及当前块的最大值、指数和、中间输出，算子会按在线Softmax公式合并两部分状态，得到更新后的最大值、指数和与未归一化输出。

该接口通常与`pypto.experimental.online_softmax`配合使用：`online_softmax`计算当前scores块的局部统计，`online_softmax_update`将当前块统计合入已有状态。最终输出通常还需要用更新后的指数和做归一化。

## 函数原型

```python
online_softmax_update(
    previous_max: Tensor,
    previous_sum: Tensor,
    previous_output: Tensor,
    current_max: Tensor,
    current_sum: Tensor,
    current_output: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| previous_max | 输入 | 历史块的列最大值。<br>支持的数据类型为：DT_FP32。<br>不支持空Tensor，支持两维。<br>Shape为[1, q_len]。 |
| previous_sum | 输入 | 历史块的列指数和。<br>支持的数据类型为：DT_FP32。<br>不支持空Tensor，支持两维。<br>Shape为[1, q_len]。 |
| previous_output | 输入 | 历史块累计的未归一化输出。<br>支持的数据类型为：DT_FP32。<br>不支持空Tensor，支持两维。<br>Shape为[head_dim, q_len]。 |
| current_max | 输入 | 当前块的列最大值，通常来自`pypto.experimental.online_softmax`。<br>支持的数据类型为：DT_FP32。<br>Shape为[1, q_len]。 |
| current_sum | 输入 | 当前块的列指数和，通常来自`pypto.experimental.online_softmax`。<br>支持的数据类型为：DT_FP32。<br>Shape为[1, q_len]。 |
| current_output | 输入 | 当前块的未归一化输出。<br>支持的数据类型为：DT_FP32。<br>Shape为[head_dim, q_len]，需要与previous_output形状一致。 |

## 返回值说明

返回三个输出Tensor：

| 返回值 | 说明 |
|--------|------|
| updated_max | 合并后的列最大值，数据类型为DT_FP32，Shape为[1, q_len]。 |
| updated_sum | 合并后的列指数和，数据类型为DT_FP32，Shape为[1, q_len]。 |
| updated_output | 合并后的未归一化输出，数据类型为DT_FP32，Shape为[head_dim, q_len]。 |

## 约束说明

1. 该接口为定制接口，不保证稳定性。
2. 所有输入Tensor数据类型仅支持 DT_FP32。
3. current_output 需要与 previous_output 形状一致。
4. 当前版本不切分第0维，要求 previous_output.shape[0] <= vec_tile[0]。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape的维度设置须与`previous_output`、`current_output`保持一致。当前版本不切分第0维，要求`previous_output.shape[0] <= vec_tile[0]`；同时最后一维Tile大小需要满足FP32的32字节对齐。

### 接口调用示例

```python
import pypto

previous_max = pypto.tensor([1, 128], pypto.DT_FP32)
previous_sum = pypto.tensor([1, 128], pypto.DT_FP32)
previous_output = pypto.tensor([128, 128], pypto.DT_FP32)
current_max = pypto.tensor([1, 128], pypto.DT_FP32)
current_sum = pypto.tensor([1, 128], pypto.DT_FP32)
current_output = pypto.tensor([128, 128], pypto.DT_FP32)

pypto.set_vec_tile_shapes(128, 64)
updated_max, updated_sum, updated_output = pypto.experimental.online_softmax_update(
    previous_max,
    previous_sum,
    previous_output,
    current_max,
    current_sum,
    current_output,
)
```
