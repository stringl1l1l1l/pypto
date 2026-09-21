# pypto.unpack

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

数据解包，将UINT8类型的输入Tensor转换为指定数据类型的输出Tensor。

## 函数原型

```python
unpack(self: Tensor, dstDataType: DataType) -> Tensor
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| self         | 输入      | 源操作数。<br>支持的数据类型为：DT_UINT8。<br>不支持空Tensor；Shape仅支持1维；支持数据格式：ND。 |
| dstDataType  | 输入      | 指定输出数据类型。<br>支持的数据类型为：DT_UINT8，DT_INT8，DT_UINT16，DT_INT16，DT_UINT32，DT_INT32，DT_UINT64，DT_INT64，DT_BF16，DT_FP16，DT_FP32。 |

## 返回值说明

返回Tensor类型。其shape为一维，数据类型为`dstDataType`指定的类型，元素数量为输入Tensor元素数除以`dstDataType`的字节数。即将输入Tensor的连续字节按`dstDataType`的字节宽度重新解释为对应类型的元素。

## 约束说明

1. InputShape、ValidShape、TileShape的大小必须能被`dstDataType`的字节数整除。
2. Shape Size不大于2147483647（即INT32_MAX）。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape维度应和输入一致。

如输入input shape为[n]，输出shape为[m]（其中 m = n / sizeof(dstDataType)），TileShape设置为[t]，则t用于切分输入轴n。

```python
pypto.set_vec_tile_shapes(16)
```

### 接口调用示例

```python
x = pypto.tensor([6, 5, 4, 3, 8, 6, 5, 1], pypto.DT_UINT8)
y = pypto.unpack(x, pypto.DT_INT32)
```

结果示例如下：

```python
输入数据x: [  6  5  4  3  8  6  5  1  ]    # DT_UINT8, shape=[8]
输出数据y: [ 0x03040506 0x01050608 ]        # DT_INT32, shape=[2]
```

说明：输入的8个UINT8元素每4个字节解释为1个INT32元素，因此8个UINT8元素解包为2个INT32元素。
