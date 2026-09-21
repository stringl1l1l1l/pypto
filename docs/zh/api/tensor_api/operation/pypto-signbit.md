# pypto.signbit

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

检查输入张量中每个元素的符号位（sign bit）是否被设置（即是否为负）。逐元素运算。

逻辑：

- 如果元素是负数（包括 −∞ 和 −0.0），返回True。
- 如果元素是正数（包括 +∞ 和 +0.0）或NaN，返回False。

## 函数原型

```python
signbit(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_INT16，DT_INT32。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape与输入Tensor一致，数据类型为DT_BOOL，其元素为输入Tensor对应元素的符号位是否被设置（True表示负数，False表示非负数）。

## 约束说明

1. TileShape与input维度保持一致；
2. 由于存在临时内存使用，当输入数据类型为DT_FP32，TileShape大小有额外约束，假设TileShape为[a,b,c,d]，那么a*b*c*d*sizeof(self) + a*b*c*d*sizeof(FP16) + a*b*c*d*sizeof(UINT8) < UB。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([-5, 0, 5, 10, -2], pypto.DT_FP32)
y = pypto.signbit(x)
```

结果示例如下：

```python
输入数据x: [-5.0, 0.0, 5.0, 10.0, -2.0]
输出数据y: [True, False, False, False, True]
```
