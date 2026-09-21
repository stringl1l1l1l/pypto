# pypto.topk

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

获取最后一个维度的前k个最大值或最小值及其对应的索引。

如果输入是向量，则在向量中找到前k个最大值或最小值及其对应的索引；如果输入是矩阵，则沿最后一个维度计算每行中前k个最大值或最小值及其对应的索引。如下图所示，对Shape为\(4, 32\)的二维矩阵进行排序，k设置为1，输出结果为\[\[32\] \[32\] \[32\] \[32\]\]。

![](../figures/nz-reduce.png)

## 函数原型

```python
topk(input: Tensor, k: int, dim: Optional[int] = None, largest: bool = True, algo: TopKAlgo = TopKAlgo.MERGE_SORT) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：<br>- MERGE_SORT：DT_FP32。<br>- RADIX_SELECT：DT_BF16，DT_FP16，DT_FP32，DT_INT64，DT_UINT64，DT_INT32，DT_UINT32，DT_INT16，DT_UINT16，DT_INT8，DT_UINT8。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| k       | 输入      | 返回元素的数量。<br>k的大小应该满足：1 <= k <= input.shape[dim]。 |
| dim     | 输入      | 指定排序的维度。<br>目前仅支持按最后一个维度排序，即dim= -1或dim= input.shape.size() - 1。 |
| largest | 输入      | 如果为True，返回最大元素。如果为False，返回最小元素。 |
| algo    | 输入      | 算法枚举类型，用以控制TopK计算的流程，具体定义为：[TopKAlgo](../datatype/TopKAlgo.md)。<br>默认为MERGE_SORT（归并排序算法）。 |

## 返回值说明

返回一个命名元组(values, indices)，其中包含input在指定维度dim下每行中最大或最小的k个元素的值和索引。

## 约束说明

1. 只支持对尾轴进行topk操作；
2. 选用MERGE_SORT算法时，TileShape尾轴需要小于22KB\(TileShape\[-1\]\*4 < 22KB\)；
3. 选用RADIX_SELECT算法时，由于存在临时内存使用，设置TileShape时需保证输入Tile、输出Tile及临时空间的总占用小于可用UB。记TileShape的次尾轴为tileH（若不存在则为1），尾轴为tileW，tileW对齐到128记为tileWAlign，各数据类型对应的临时空间如下：

   | 输入数据类型 | 临时空间大小（字节） |
   |--------------|----------------------|
   | DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16 | `tileH * max(6 * tileWAlign + max(3072, 8 * tileWAlign), 22 * tileWAlign)` |
   | DT_FP32、DT_INT32、DT_UINT32 | `tileH * max(12 * tileWAlign + max(3072, 8 * tileWAlign), 26 * tileWAlign)` |
   | DT_INT64、DT_UINT64 | `tileH * max(22 * tileWAlign + max(3072, 8 * tileWAlign), 34 * tileWAlign)` |

   表格中的max\(a,b\)表示取a和b中的最大值。
4. 选用RADIX_SELECT算法时，尾轴不可切分，TileShape\[-1\]必须大于等于input.shape\[-1\]；
5. k <= TileShape\[-1\] && k <= input.shape\[-1\]；
6. RADIX_SELECT算法在不同型号的支持度：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：支持
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：不支持
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：不支持
   <!-- end id6 -->
7. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入input一致。

示例1：输入input shape为[m, n, p]，dim为2，largest为True，输出为[m, n, k]，TileShape设置为[m1, n1, p1]，则m1，n1，p1分别用于切分m，n，p轴。p1必须大于等于k，k轴不支持切分，必须保证全载。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.MERGE_SORT)
```

结果示例如下：

```python
输入数据x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
输出数据y[0]: [[3.0 2.0],
               [3.0 2.0]]
输出数据y[1]: [[2, 1],
               [2, 1]]
```
