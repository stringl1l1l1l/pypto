# pypto.deinterleave

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

将交织数据反交织为两个输出 Tensor。第一个输出 Tensor 接收交织流偶数位置的元素，第二个输出 Tensor 接收交织流奇数位置的元素。

`pypto.deinterleave` 支持两种输入形式：

- **双输入形式**：`pypto.deinterleave(input, other)`。`input` 保存交织流前半部分，`other` 保存交织流后半部分，接口先重构完整交织流，再按偶数位置和奇数位置拆分。
- **单输入形式**：`pypto.deinterleave(input)`。`input` 保存完整交织流，接口直接按最后一个维度的偶数位置和奇数位置拆分。输出 Tensor 的最后一个维度为输入 Tensor 最后一个维度的一半。

`pypto.deinterleave` 是 `pypto.interleave` 的逆操作。

## 数学语义

### 双输入形式

给定 `input` 和 `other`，先沿最后一个维度拼接为交织流：

```text
combined[j] = input[j],                         0 <= j < cols
combined[j] = other[j - cols],                  cols <= j < 2 * cols
```

再反交织：

```text
out0[k] = combined[2 * k]
out1[k] = combined[2 * k + 1]
```

### 单输入形式

给定保存完整交织流的 `input`：

```text
out0[k] = input[2 * k]
out1[k] = input[2 * k + 1]
```

## 函数原型

```python
deinterleave(input: Tensor, other: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ---- |
| input  | 输入      | 源操作数。双输入形式下，表示交织流的前半部分；单输入形式下，表示完整交织流。支持的类型为 Tensor。 |
| other  | 输入      | 可选参数。双输入形式下，表示交织流的后半部分，Shape 和数据类型需要与 `input` 一致。若不传入，则使用单输入形式。 |

## 返回值说明

返回一个二元组 `(out0, out1)`。

- `out0` 保存交织流偶数位置的元素。
- `out1` 保存交织流奇数位置的元素。
- 双输入形式下，`out0` 和 `out1` 的 Shape、数据类型与输入 Tensor 一致。
- 单输入形式下，`out0` 和 `out1` 的数据类型与 `input` 一致；除最后一个维度外，Shape 与 `input` 一致，最后一个维度为 `input.shape[-1] / 2`。

## 约束说明

1. 支持的数据类型为：`DT_INT8`、`DT_UINT8`、`DT_INT16`、`DT_UINT16`、`DT_INT32`、`DT_UINT32`、`DT_FP16`、`DT_FP32`、`DT_BF16`。
2. 当前支持 1 到 4 维 Tensor。
3. 双输入形式下，`input` 和 `other` 的数据类型、维度数、Shape 必须一致，且最后一个维度的 Shape 必须为偶数。
4. 单输入形式下，`input` 的最后一个维度 Shape 必须为偶数。
5. 当 TileShape 有效配置时，TileShape 维度应与输入 Tensor 维度一致。
6. 双输入形式下，TileShape 最后一维必须与输入 Tensor 的 Shape 最后一维相等，即最后一维不能切分；其他维可按切分需求设置。
7. 单输入形式下，TileShape 最后一维可以切分，但必须为偶数，以保证每个 Tile 内的偶数位置和奇数位置元素成对；其他维可按切分需求设置。
8. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape 设置示例

双输入形式下，输入 `input`、`other` 和输出 `out0`、`out1` 的 Shape 均为 `[m, n]`，TileShape 设置为 `[m1, n1]`，则 `m1`、`n1` 分别用于切分 `m`、`n` 轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

单输入形式下，输入 `input` 的 Shape 为 `[m, 2 * n]`，输出 `out0`、`out1` 的 Shape 均为 `[m, n]`。TileShape 最后一维可以小于输入 Tensor 的最后一维，例如以下配置会沿最后一维分 Tile 执行：

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 双输入接口调用示例

```python
input = pypto.tensor([2, 4], pypto.DT_FP32)
other = pypto.tensor([2, 4], pypto.DT_FP32)
out0, out1 = pypto.deinterleave(input, other)
```

结果示例如下：

```python
输入数据 input: [[0.0, 10.0, 1.0, 11.0],
               [4.0, 14.0, 5.0, 15.0]]
输入数据 other: [[2.0, 12.0, 3.0, 13.0],
               [6.0, 16.0, 7.0, 17.0]]

输出数据 out0: [[0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0]]
输出数据 out1: [[10.0, 11.0, 12.0, 13.0],
               [14.0, 15.0, 16.0, 17.0]]
```

### 单输入接口调用示例

```python
input = pypto.tensor([2, 8], pypto.DT_FP32)
out0, out1 = pypto.deinterleave(input)
```

结果示例如下：

```python
输入数据 input: [[0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0],
               [4.0, 14.0, 5.0, 15.0, 6.0, 16.0, 7.0, 17.0]]

输出数据 out0: [[0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0]]
输出数据 out1: [[10.0, 11.0, 12.0, 13.0],
               [14.0, 15.0, 16.0, 17.0]]
```

## 相关接口

- [pypto.interleave](pypto-interleave.md)：将两个 Tensor 交织为交替的偶/奇元素流，并拆分为两个输出 Tensor。
