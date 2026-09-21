# pypto.assemble

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

以offsets指定的out索引位置为基准，将输入Tensor input赋值到输出Tensor out的对应区域。

## 函数原型

```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor, parallel: bool = False) -> None

assemble(inputs: List[Tuple[Tensor, List[Union[int, SymbolicScalar]]]], out: Tensor, parallel: bool = False) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：PyPto支持的数据类型。<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| inputs   | 输入      | 源操作数和输出偏移组成的Tuple列表。<br>单个支持的数据类型为：PyPto支持的数据类型。<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| offsets | 输入      | 相对于目标输出的偏移。<br>需要保证offsets小于out的Shape。          |
| out     | 输出      | 目的操作数，需要和input的维度数量一致。<br>支持的数据类型为：PyPto支持的数据类型。<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| parallel | 输入      | 是否允许并行写回。默认值为False；当写回区域互不重叠、可安全并行时，应显式传入`parallel=True`。 |

## 返回值说明

无返回值，会直接对out进行修改。

## 约束说明

1. 输出Tensor out的valid shape需由用户在调用assemble前确保正确，该接口不会自动推导。
2. 输入张量input和输出张量out的维度数量需要一致。
3. 当多个assemble对同一out的重叠区域存在写后写依赖，且这些写回分布在不同loop迭代或不同function中时，默认`parallel=False`已保证写回按依赖顺序串行执行，框架会在对应outcast上标记`NORMAL`，供后续调度按串行写处理。若写回区域互不重叠、可安全并行，应显式传入`parallel=True`。

## 调用示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out)

y = pypto.tensor([2, 2], pypto.DT_FP32)
pypto.assemble([(x, offsets), (y, [2, 2])], out)
```

结果示例如下：

```python
输出数据x: [[1, 1]
           [1, 1]]
输入数据out: [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
输出数据out: [[1, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
输出数据out1: [[1, 1, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 1, 1]]
```

### 跨loop / 跨function的串行写回

当同一输出在不同loop迭代或不同function之间存在写后写依赖、必须串行assemble时，默认行为即为串行写回；也可显式传入`parallel=False`：

```python
# 跨loop：后一次写依赖前一次写的结果，默认串行
for i in pypto.loop(0, n, name="SEQ_WRITE"):
    tile = ...
    pypto.assemble(tile, [i * tile_m, 0], out)

# 跨function：下游function继续写同一out的重叠区域时，同样默认串行
pypto.assemble(partial, [offset_m, offset_n], out)
```

说明：

- 单tensor与批量多源assemble在未传`parallel`时均默认为False；需要并行写回时显式传入`parallel=True`。
- 若各次assemble写回区域不重叠且需并行调度，应设置`parallel=True`。
