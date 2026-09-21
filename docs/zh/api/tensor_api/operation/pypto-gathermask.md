# pypto.gathermask

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

根据PatternMode选择的内置Mask，对输入Tensor对应Bit位为1的位置组成输出Tensor，Bit位为0的值直接丢弃。PatternMode有7种模式：

- PatternMode=1，尾轴每两个元素取第一个元素
- PatternMode=2，尾轴每两个元素取第二个元素
- PatternMode=3，尾轴每四个元素取第一个元素
- PatternMode=4，尾轴每四个元素取第二个元素
- PatternMode=5，尾轴每四个元素取第三个元素
- PatternMode=6，尾轴每四个元素取第四个元素
- PatternMode=7，尾轴取全部元素

## 函数原型

```python
gathermask(self: Tensor, pattern_mode: int) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_FP16，DT_BF16，DT_FP32。<br>不支持空Tensor，Shape支持1-4维，且shape size不大于2147483647（即INT32_MAX）。 |
| pattern_mode | 输入      | 控制参数。<br>int类型，取值范围为：1~7。 |

## 返回值说明

返回输出Tensor，输出Tensor数据类型与self数据类型一致，输出Tensor的Shape如下：

- pattern_mode <= 2，输出shape尾轴=self.shape尾轴/2，其它轴和输入shape一致；
- 2 < pattern_mode < 7，输出shape尾轴=self.shape尾轴/4，其它轴和输入shape一致；
- pattern_mode = 7，输出shape=输入shape

## 约束说明

1. 当1 <= pattern_mode <= 2时：
   - self.shape尾轴必须能被2整除
   - tileshape尾轴必须是2的整倍数
   - viewshape尾轴必须是2的整倍数
   - self.shape尾轴不做view切分
2. 当3 <= pattern_mode <= 6时：
   - self.shape尾轴必须是4的整倍数
   - tileshape尾轴必须是4的整倍数
   - viewshape尾轴必须是4的整倍数
   - self.shape尾轴不做view切分
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入self为[x, y, z]，pattern_mode为1，输出为[x, y, z/2]，TileShape设置为[x1, y1, 2*z1]，则x1，y1，2\*z1分别用于切分x，y，z轴。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([3, 6], pypto.DT_INT32)        # shape (3, 6)
pattern_mode = 1
y = pypto.gathermask(x, pattern_mode)
```

结果示例如下：

```python
输入数据x: [[0,  1,  2,  3,  4,  5],
             [6,  7,  8,  9,  10,  11],
             [12,  13,  14,  15,  16,  17]]
     pattern_mode: 1
输出数据y: [[0,  2,  4],
             [6,  8,  10],
             [12,  14,  16]]
```
