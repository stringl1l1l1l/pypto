# pypto.permute

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

返回一个Tensor，该Tensor是输入Tensor的转置版本。根据指定的维度顺序重新排列输入张量的维度，该算子不改变张量中元素的总数和内容，只改变维度的排列方式。

## 函数原型

```python
permute(input: Tensor, perm: list[int]) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-5维；Shape Size不大于2147483647（即INT32_MAX）。 |
| perm    | 输入      | 维度顺序列表。必须是一个包含所有维度索引的排列，长度与输入张量的维度数相同。每个维度索引在0到ShapeSize-1范围内且不重复。 |

## 返回值说明

返回一个与输入数据类型一致的Tensor，其维度顺序按照perm指定的顺序重新排列。

## 约束说明

1. 输入Tensor和输出Tensor的数据类型必须相同。
2. 64位整型格式限制：DT_INT64、DT_UINT64不支持NZ（Fractal-Z）格式，仅支持ND格式。
3. TileShape尾轴32字节对齐约束（仅无需实际排列的场景）：当permute判定为无需实际排列、直接返回输入Tensor时（即输入为1维Tensor，或perm为恒等排列），TileShape的最后一维的字节数必须对齐到32字节（BLOCK_SIZE）。即：`TileShape最后一维元素数 × sizeof(数据类型) % 32 == 0`。

   说明：此场景直接返回输入Tensor（`return self`），但后续流程中的TILE_REGISTER_COPY仍会校验TileShape尾轴的32字节对齐。若未满足此约束，将报错：`CHECK FAILED: lastDimBytes % BLOCK_SIZE == 0`。

4. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64，DT_BOOL，DT_FP8E4M3，DT_FP8E5M2，DT_HF8，DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64，DT_BOOL
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64，DT_BOOL
   <!-- end id6 -->
5. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。TileShape的维度应与输入input一致。

示例：输入input shape为 [2, 3, 4]，目标排列为 [2, 0, 1]，TileShape可设置为 [2, 3, 4] 或根据切分需求设置合适的值。

```python
pypto.set_vec_tile_shapes(2, 3, 4)
```

### 接口调用示例

```python
x = pypto.tensor([1, 2, 3, 4], pypto.DT_FP32)
perm = [3, 1, 0, 2]
y = pypto.permute(x, perm)
```

结果示例如下：

```python
输入数据x: [[[[ 0.9586, -0.4325,  0.7582, -2.6209],
              [ 1.0931, -0.3324, -2.3653, -0.0324],
              [ 1.6083,  1.3619, -0.1481,  0.4394]],

             [[ 0.2353, -0.7177, -0.4954,  0.4158],
              [-0.9788, -1.4224,  0.2558,  1.5322],
              [-0.6645,  2.1023,  0.8968,  0.8690]]]],

输出数据y: [[[[ 0.9586,  1.0931,  1.6083]],
             [[ 0.2353, -0.9788, -0.6645]]],

            [[[-0.4325, -0.3324,  1.3619]],
             [[-0.7177, -1.4224,  2.1023]]],

            [[[ 0.7582, -2.3653, -0.1481]],
             [[-0.4954,  0.2558,  0.8968]]],

            [[[-2.6209, -0.0324,  0.4394]],
             [[ 0.4158,  1.5322,  0.8690]]]]
```
