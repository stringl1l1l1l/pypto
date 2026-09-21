# pypto.experimental.gather\_in\_l1

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

该接口为定制接口，不保证稳定性。

从GM上的Tensor离散搬运指定行的数据，同时每行搬运前size个数据至L1。

## 函数原型

```python
gather_in_l1(src: Tensor, indices: Tensor, block_table: Tensor, block_size: int,
                 size: int, is_b_matrix: bool, is_trans: bool) -> Tensor
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| src          | 输入      | 源操作数。<br>支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8。<br>不支持空Tensor，支持两维。 |
| indices      | 输入      | 源操作数的行偏移。<br>支持的数据类型为：DT_INT32，DT_INT64。<br>不支持空Tensor，支持两维。<br>Shape形状为[1,n]。 |
| block_table  | 输入      | 页表。<br>支持的数据类型为DT_INT32。<br>不支持空Tensor，支持两维。<br>在实际使用中表示为Page Attention中的页表，形状为[1,block_table_size]，其中block_table_size表示页表的长度。 |
| block_size   | 输入      | 源操作数。<br>int类型。<br>表示Page Attention中一个块可以放多少个token。 |
| size         | 输入      | 每行搬运的数据数。<br>数据数要小于源操作数的列数。 |
| is_b_matrix  | 输入      | 搬运后的结果，即输出Tensor是否作为matmul的B矩阵。 |
| is_trans     | 输入      | 搬运后的结果，即输出Tensor是否转置。 |

## 返回值说明

返回输出Tensor

## 约束说明

本接口为试验特性，后续版本可能会存在变更，不支持应用于生产环境中。

## 调用示例

```python
src = pypto.tensor([16, 32], pypto.DT_FP32, "tensor_src")
offset = pypto.tensor([1, 32], pypto.DT_INT32, "tensor_offset")
block_table = pypto.tensor([1, 4], pypto.DT_INT32, "block_table")
block_size = 2
size = 16
is_b_matrix = False
is_trans = False
out = pypto.experimental.gather_in_l1(src, offset, block_table, block_size, size, is_b_matrix, is_trans)
```
