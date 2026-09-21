# pypto.index\_add\_

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

将source的每一块数据乘以缩放因子alpha（默认为1）加到input的相应数据块上，其中索引和数据块方向由index和dim指定。例如，
$$ input[index[i], :, :] += alpha * source[i, :, :],\ if \ dim == 0, \\
    input[:, index[i], :] += alpha * source[:, i, :],\ if \ dim == 1,\\
    input[:, :, index[i]] += alpha * source[:, :, i],\ if \ dim == 2.$$

## 函数原型

```python
index_add_(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32。<br>不支持空Tensor；Shape仅支持1-5维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | int类型，加法作用到input的维度；<br>支持任意不超过input维数的值，详见约束说明。 |
| index   | 输入      | 源操作数，值代表input所在dim轴的索引；<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT32，DT_INT64；<br>不支持空Tensor，Shape只支持1维，索引与source的dim轴索引一一对应，Shape大小与source所在dim轴的Shape大小相同。 |
| source  | 输入      | 需要加到input的源操作数；<br>支持的类型为：Tensor。<br>Tensor的数据类型与input相同。<br>Shape支持1-5维，所在dim轴的Shape大小与index相同，其他维度的Shape大小与input相同。 |
| alpha   | 输入      | 标量，关键字参数；<br>表示累加时的缩放因子，默认为1。 |

## 返回值说明

原地操作返回input

## 约束说明

1. index必须是整数类型（DT\_INT32或DT\_INT64），值不超过input在dim维度上的Shape大小，维数为1，Shape大小与source所在dim轴的Shape大小相同；

2. dim为int类型，取值范围：$-input.dim\leq dim < input.dim$；

3. input和source的数据类型和维数均相同；

4. input.shape和source.shape的非dim轴ViewShape不可切，即 $ViewShape[i] \geq input.shape[i]=source.shape[i]，i \ne dim$；

5. TileShape的维度与source相同，只用来切分source和index，所有输入和输出的TileShape大小总和不能超过UB内存的大小。
6. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

如输入input为[m, n, p]，dim为1，输入source为[m, t, p]，输入index为[t]，输出为[m, n, p]，TileShape设置为[m1, t1, p1]，则m1，t1，p1分别用于切分source的m，t，p轴。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
source = pypto.tensor([3, 3], pypto.DT_INT32)   # shape (3, 3)
index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
dim = 0
# use alpha
y = pypto.index_add_(x, dim, index, source, alpha=1)
# not use alpha
y = pypto.index_add_(x, dim, index, source)
```

结果示例如下：

```python
输入数据x:   [[0 0 0],
               [0 0 0]]
      source: [[1 1 1],
               [1 1 1],
               [1 1 1]]
      index:   [0 1 0]
输出数据y:   [[2 2 2],
               [1 1 1]]               # shape (2, 3)
```
