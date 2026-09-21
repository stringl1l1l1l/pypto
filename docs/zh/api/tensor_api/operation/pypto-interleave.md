# pypto.interleave

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

将两个输入 Tensor（`input` 和 `other`）按最后一个维度逐元素交织，并将交织后的数据流按中点拆分为两个输出 Tensor。

对每一行最后一维元素，先构造交织流：

```text
interleaved[2 * k]     = input[k]
interleaved[2 * k + 1] = other[k]
```

然后将交织流的前半部分写入第一个输出 Tensor，后半部分写入第二个输出 Tensor。`pypto.interleave` 是 `pypto.deinterleave` 的逆操作。

## 函数原型

```python
interleave(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ---- |
| input  | 输入      | 第一个源操作数。支持的类型为 Tensor。 |
| other  | 输入      | 第二个源操作数。支持的类型为 Tensor。`other` 的 Shape、数据类型需要与 `input` 一致。 |

## 返回值说明

返回一个二元组 `(out0, out1)`。

- `out0` 保存交织流的前半部分。
- `out1` 保存交织流的后半部分。
- `out0` 和 `out1` 的 Shape、数据类型与 `input` 一致。

## 约束说明

1. `input` 和 `other` 的数据类型、维度数、Shape 必须一致。
2. 支持的数据类型为：`DT_INT8`、`DT_UINT8`、`DT_INT16`、`DT_UINT16`、`DT_INT32`、`DT_UINT32`、`DT_FP16`、`DT_FP32`、`DT_BF16`。
3. 当前支持 1 到 4 维 Tensor。
4. 最后一个维度的 Shape 必须为偶数。
5. 当 TileShape 有效配置时，TileShape 维度应与输入 Tensor 维度一致，且最后一维必须与输入 Tensor 的 Shape 最后一维相等；其他维可按切分需求设置。
6. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape 设置示例

输入 `input` 和 `other` 的 Shape 为 `[m, n]`，输出 `out0` 和 `out1` 的 Shape 均为 `[m, n]`，TileShape 设置为 `[m1, n1]`，则 `m1`、`n1` 分别用于切分 `m`、`n` 轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
input = pypto.tensor([2, 4], pypto.DT_FP32)
other = pypto.tensor([2, 4], pypto.DT_FP32)
out0, out1 = pypto.interleave(input, other)
```

结果示例如下：

```python
输入数据 input: [[0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0]]
输入数据 other: [[10.0, 11.0, 12.0, 13.0],
               [14.0, 15.0, 16.0, 17.0]]

输出数据 out0: [[0.0, 10.0, 1.0, 11.0],
               [4.0, 14.0, 5.0, 15.0]]
输出数据 out1: [[2.0, 12.0, 3.0, 13.0],
               [6.0, 16.0, 7.0, 17.0]]
```

## 相关接口

- [pypto.deinterleave](pypto-deinterleave.md)：将交织流反交织回偶数位置和奇数位置的元素流。
