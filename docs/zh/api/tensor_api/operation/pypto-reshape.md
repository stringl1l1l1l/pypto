# pypto.reshape

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

改变Tensor形状，改变valid\_shape部分的形状\(Shape\)

## 注意事项

- **静态shape支持`-1`**：当tensor所有轴都是静态维度时，shape参数支持使用`-1`自动推导一个维度
- **动态shape不支持`-1`**：当tensor有轴标注为`pypto.DYNAMIC`时，shape参数不能使用`-1`，必须显式指定所有维度值，从动态轴`tensor.shape`获取的维度是SymbolicScalar类型，可用于reshape的shape参数
- **输入validshape不会自动继承**：若输入Tensor带有validshape，且希望输出也保留validshape，需要用户自行计算出输出的validshape，并通过`valid_shape`参数传入；未传入时，reshape不会根据输入validshape自动推导输出validshape
- **推荐使用inplace参数**：当满足inplace的约束说明时，设置`inplace=True`可以避免额外的数据搬移

## 函数原型

```python
reshape(input: Tensor,shape: List[int],*,valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, inplace: bool = False) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。<br>支持的数据类型为：PyPTO支持的数据类型<br>不支持空Tensor。 |
| shape       | 输入      | 目标Shape。<br>- **静态shape**：支持使用`-1`自动推导一个维度。<br>- **动态shape**：不支持`-1`，必须显式指定所有维度值。维度值可以是具体整数或SymbolicScalar（从动态轴获取）。 |
| valid_shape | 输入      | 输出Tensor的有效数据的Shape。<br>输入带有validshape时，框架不会自动推导输出validshape；若需要输出保留validshape，须由用户计算后通过本参数传入。 |
| inplace     | 输入      | 是否为inplace，默认为False；参数为True时，不会为输出申请新地址； |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，形状\(Shape\)为输入参数指定的shape。

## 约束说明

1. view生成的张量执行reshape，仅允许inplace为False。
2. inplace为True时，reshape通常需单独置于loop (1)中，无其他类型的operation并列时，可省略loop (1)，框架自动补齐，见示例2。
3. inplace为True的输出，不可作为函数最终输出。
4. inplace=False仅适配静态shape；inplace=True兼容静态shape和动态shape。

## 调用示例

示例1：

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.reshape(x, [4, 1], [2, 1])
z = pypto.add(y, 1.0)
```

结果示例如下：

```python
输入数据x: [[1, 2],
            [3, 4]]
输出数据y: [[1],
            [2],
            [3],
            [4]]
输出数据z: [[2],
            [3],
            [4],
            [5]]
```

示例2：

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
# reshape(..., inplace=True)单独在loop(1)内。loop(1)可省略
for _ in pypto.loop(1, name="reshape_inplace", idx_name="tmp_loop"):
    x_1 = x.reshape([4], inplace=True)
for _ in pypto.loop(1, name="loop", idx_name="loop"):
    y = pypto.add(x_1, 1.0)
```

结果示例如下：

```python
输入数据x: [[1, 2],
            [3, 4]]
输出数据y: [2, 3, 4, 5]
```
