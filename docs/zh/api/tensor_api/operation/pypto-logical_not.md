# pypto.logical\_not

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

输入Tensor中的0对应转换为True，非0值转换为False。

## 函数原型

```python
logical_not(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型为DT\_BOOL，Shape与源操作数input Shape相同。

## 约束说明

1. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_FP32，DT_FP16，DT_BF16，DT_BOOL，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_FP32，DT_FP16，DT_BF16，DT_BOOL，DT_INT8，DT_UINT8。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_FP32，DT_FP16，DT_BF16，DT_BOOL，DT_INT8，DT_UINT8。
   <!-- end id6 -->
2. TileShape与input维度保持一致；
3. 由于存在临时内存使用，假设TileShape为$[a,b,c,d]$，记$N=a \cdot b \cdot c \cdot d$。当输入数据类型为DT_FP32时，TileShape大小应满足$N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 20.25\text{KB} < \text{UB}$；其他基础支持类型应满足$N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 12.54\text{KB} < \text{UB}$。
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：输入数据类型为DT_INT16、DT_UINT16、DT_INT32、DT_UINT32或DT_INT64时，采用整数计算实现。其中，DT_INT16/DT_UINT16需要4KB临时内存，DT_INT32/DT_UINT32需要8KB临时内存，DT_INT64需要16KB临时内存。上述类型的TileShape大小应满足$N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 2048 \cdot \text{sizeof}(input) < \text{UB}$。
   <!-- end id7 -->
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
a = pypto.tensor([5], pypto.DT_INT32)
out = pypto.logical_not(a)
```

结果示例如下：

```python
输入数据x: [0 1 2 3 4]
输出数据y: [True False False False False]
```
