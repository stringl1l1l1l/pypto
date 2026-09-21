# pypto.pack

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

将输入Tensor铺平为一维，并将其原始字节解释为uint8元素。

## 函数原型

```python
pack(self: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self    | 输入      | 源操作数。<br>支持的数据类型为：DT_BF16，DT_FP16，DT_FP32，DT_INT64，DT_UINT64，DT_INT32，DT_UINT32，DT_INT16，DT_UINT16，DT_INT8，DT_UINT8。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其shape为一维，元素数量与输入Tensor元素所占字节数相同，其元素为输入Tensor对应元素的uint8表示。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入一致。

如输入input shape为[m, n]，输出为[ k ]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([1, 2], pypto.DT_INT32)
y = pypto.pack(x)
```

结果示例如下：

```python
输入数据x: [[ 0x03040506 0x01050608]]
输出数据y: [  6  5  4  3  8  6  5  1  ]
```
