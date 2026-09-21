# pypto.tril

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

返回二维张量或者一批张量的下三角部分。结果张量的其他元素被设置为0。

## 函数原型

```python
tril(input: Tensor, diagonal: SymInt = 0) -> Tensor
```

## 参数说明

| 参数名   | 输入/输出 | 说明                                                                                                                                                                                                                        |
| -------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input    | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持2-5维；Shape Size不大于2147483647（即INT32_MAX）。 |
| diagonal | 输入      | 对角线偏移量，默认为0（主对角线）。<br>SymInt类型。                                                                                                                                                    |

## 返回值说明

输出Shape、数据类型与输入input一致的Tensor。

## 约束说明

1. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32，DT_UINT16，DT_UINT32，DT_INT64，DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32
   <!-- end id6 -->

2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度数应和输出张量维度数一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([3, 3], pypto.data_type.DT_INT32)        # shape (3, 3)
diagonal = 0
out = pypto.tril(x, diagonal)
```

结果示例如下：

```python
输入数据x :[[1 2 3],
             [4 5 6],
             [7 8 9]]
输出数据out:[[1 0 0],
             [4 5 0],
             [7 8 9]]                             # shape (3, 3)
```
