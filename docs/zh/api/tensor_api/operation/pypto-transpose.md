# pypto.transpose

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

返回一个Tensor，该Tensor是输入Tensor的转置版本。指定的维度dim0和dim1将被交换。

## 函数原型

```python
transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-5维；Shape Size不大于2147483647（即INT32_MAX）。<br>算子对不同Shape支持不同，详见约束说明。 |
| dim0    | 输入      | 输入参数，要交换的第一个维度的索引，从0开始计数。 |
| dim1    | 输入      | 输入参数，要交换的第二个维度的索引，从0开始计数。 |

## 返回值说明

返回一个与输入数据类型一致的Tensor，其中dim0与dim1的维度位置被对调。

## 约束说明

1. TileShape和输入input维度一致，用于切分input。

2. 输入维度dim0，dim1的取值范围为：-D ≤ dim ≤ D-1，其中D为input的维度数。

3. 当前Transpose实现存在约束，只能支持以下场景转置：

   - 2维：任意轴
   - 3维：任意轴
   - 4维：支持：0轴和2轴，1轴和3轴，2轴和3轴，1轴和2轴，不支持：0轴和3轴，0轴和1轴
   - 5维：支持：3轴和4轴，其他不支持
   - 无需实际转置的场景直接支持：当dim0和dim1相同，或dim0和dim1对应的输入shape维度均为1时，transpose结果与输入等价，不受前述4维/5维轴组合约束限制。

4. TileShape尾轴32字节对齐约束（仅无需实际转置的场景）：当transpose判定为无需实际转置、直接返回输入Tensor时（即dim0==dim1，或两轴shape均为1），TileShape的最后一维的字节数必须对齐到32字节（BLOCK_SIZE）。即：`TileShape最后一维元素数 × sizeof(数据类型) % 32 == 0`。

    说明：此场景直接返回输入Tensor（`return self`），但后续流程中的TILE_REGISTER_COPY仍会校验TileShape尾轴的32字节对齐。若未满足此约束，将报错：`CHECK FAILED: lastDimBytes % BLOCK_SIZE == 0`。

5. 涉及尾轴转置的场景，需要预留一块临时空间，用来搬运。

    示例：

    input：\[a, b, c, d\]  TileShape为\[t0, t1, t2, t3\] 数据类型为DT\_FP32

    dim0: 2

    dim1: 3

    预留的临时空间为：t0 \* t1 \* align\(t2,16\) \* align\(t3,32 / sizeof\(DT\_FP32\)\)

6. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_FP32，DT_INT32，DT_UINT32，DT_HF8，DT_FP8E4M3，DT_FP8E5M2，DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_FP32，DT_INT32，DT_UINT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_FP32，DT_INT32，DT_UINT32
   <!-- end id6 -->
7. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入input一致。

示例1：输入input shape为[m, n, p]，dim0为1，dim1为2，输出为[m, p, n]，TileShape设置为[m1, n1, p1]，则m1，n1，p1分别用于切分m，n，p轴。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.transpose(x, 0, 1)
```

结果示例如下：

```python
输入数据x: [[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]]
输出数据y: [[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [0.5809,  0.4942]]
```
