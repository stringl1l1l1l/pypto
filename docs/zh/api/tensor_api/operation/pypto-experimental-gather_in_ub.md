# pypto.experimental.gather\_in\_ub

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

该接口为定制接口，约束较多。不保证稳定性。

该算子支撑稀疏注意力机制，能力为将选中的token的kv cache从GM（Global Memory）加载到UB（Unified Buffer）中，支持Page Attention。

## 函数原型

```python
gather_in_ub(param: Tensor, indices: Tensor, block_table: Tensor,
                 block_size: int, axis: int) -> Tensor
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| param        | 输入      | 源操作数。<br>不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor，支持两维。<br>在实际使用中表示kv cache，形状为[token_size,hidden_dim]。 |
| indices      | 输入      | 源操作数。<br>支持的数据类型为DT_INT32。<br>不支持空Tensor，支持两维。<br>在实际使用中表示为topk输出结果，形状为[1,k]。 |
| block_table  | 输入      | 源操作数。<br>支持的数据类型为DT_INT32。<br>不支持空Tensor，支持两维。<br>在实际使用中表示为Page Attention中的页表，形状为[1,block_table_size]，其中block_table_size表示页表的长度。 |
| block_size   | 输入      | 源操作数。<br>int类型。<br>表示Page Attention中一个块可以放多少个token。 |
| axis         | 输入      | 源操作数。<br>int类型。<br>只支持-2轴。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和param相同，Shape为\[k, hidden\_dim\]，即选中token kv cache。

## 约束说明

1. 该接口为定制接口，不保证稳定性。
2. param、indices、block_table 均不支持空Tensor，仅支持两维。
3. indices Shape 必须为 [1, k]。
4. axis 参数仅支持 -2。
5. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_FP16，DT_FP32，DT_BF16，DT_BOOL，DT_FP8E4M3，DT_FP8E5M2，DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_FP16，DT_FP32，DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT8，DT_INT16，DT_INT32，DT_UINT8，DT_UINT16，DT_UINT32，DT_FP16，DT_FP32，DT_BF16
   <!-- end id6 -->

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape的维度设置须与输出张量保持一致，用于控制输出Tile块的大小。

以输入$ param[token_size,hidden_dim]$ 、索引 $indices[1,k]$ 、轴 $\text{axis}=-2$ 、输出 $output[k,hidden_dim]$  为例：

设TileShape为$[k_1, hidden_dim_1]$，该配置直接作用于输出output的各维度，同时映射至输入与索引。其中 $k_1$ 切分indices的k维，$ hidden_dim_1$ 切分param的特征维 $hidden_dim$ 。Tile内存占用须满足约束 $b_1 \cdot k_1 \cdot hidden_dim_1 \cdot \text{sizeof}(\mathbf{output}) < \text{UB_Size}$

### 接口调用示例

![](../figures/gather_in_ub.png)

考虑以上场景，indices为topk结果，block\_table为Page Attention的页表，param为kv cache，block\_size为2。最终的结果是将token的kv cache收集起来。

以token id 4为例（在图中标红），根据block_size计算出实际偏移：

blockIdx = 4 / 2; //计算对应的逻辑块，第2个逻辑块

tail = 4 % 2;        //计算块内偏移，偏移为0

slcBlockIdx = block_table\[0, blockIdxInBatch\];  //查表，得到该块实际偏移，对应第1个物理块

offsets = slcBlockIdx \* block_size + tail;//计算出实际的偏移，为2

对数据进行搬运

```python
param = pypto.tensor([6, 4], pypto.DT_FP32)
indices = pypto.tensor([1, 3], pypto.DT_INT32)
block_table = pypto.tensor([1, 3], pypto.DT_INT32)
block_size = 2
axis = -2
result = pypto.experimental.gather_in_ub(param , indices , block_table, block_size , axis)
```

结果示例如下：

```python
输入数据param :
[
  # token 0
  [  0,  1,  2,  3],
  # token 1
  [ 10, 11, 12, 13],
  # token 2
  [ 20, 21, 22, 23],
  # token 3
  [ 30, 31, 32, 33],
  # token 4
  [ 40, 41, 42, 43],
  # token 5
  [ 50, 51, 52, 53],
]
输入数据indices : [0, 4, 3]
输入数据block_table : [0, 2, 1]
输出数据out:
[
   [  0,  1,  2,  3],
   [ 20, 21, 22, 23],
   [ 50, 51, 52, 53],
]
```
