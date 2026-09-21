# pypto.argsort

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

获取输入在指定轴按照升序或者降序进行排序后的索引。

## 函数原型

```python
argsort(input: Tensor, dim: Optional[int]=None, descending: bool=True) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定排序的维度。<br>支持负索引,取值范围为[-input.dim, input.dim - 1]; 默认值为None,表示最后一个轴。四维输入不支持第0轴(即dim=0或者dim=-4)。|
| descending | 输入      | 如果为True，按降序返回索引。如果为False，按升序返回索引。 |

## 返回值说明

返回一个Tensor，为输入按照descending在dim轴进行排序后的索引。

## 约束说明

1. 当前不支持ViewShape在dim轴切分，即要求ViewShape[dim] = InputShape[dim]。
2. 当前仅支持TileShape在dim轴上为32的倍数，即要求TileShape[dim] % 32 = 0。
3. 当Shape较大场景 $(\frac{TileShape\ Size}{TileShape[dim]} * CeilAlign(ViewShape[dim], 32) >= 6144)$ 时，排序轴的Tile切分数量小于128。
4. 对于四维输入，暂不支持在第0轴上进行排序。
5. 排序时遇到相同值时，采用稳定排序返回对应的索引。
6. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入input一致。

如输入input shape为[m, n, p]，dim为2，descending为True，输出为[m, n, p]，TileShape设置为[m1, n1, p1]，则m1，n1，p1分别用于切分m，n，p轴。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argsort(x, -1, True)
```

结果示例如下：

```python
输入数据x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
输出数据y: [[2, 1, 0],
            [2, 1, 0]]
```
