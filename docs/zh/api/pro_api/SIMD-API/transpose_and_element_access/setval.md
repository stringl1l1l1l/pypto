# pypto_pro.language.setval

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

向Tile或Tensor中指定线性偏移位置写入一个标量值，与[pypto_pro.language.getval](getval.md)配合使用。根据第一个参数的类型（Tile或Tensor）选择对应实现。

## 函数原型

```python
pypto_pro.language.setval(container: Tile | Tensor, offset: int, value: Scalar) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| container | 输入/输出 | 目的操作数，Tile或Tensor类型，向其中写入单个元素。Tile必须位于UB；Tensor为GM Tensor。支持可参与标量表达式的整型或浮点类型，不支持DT_FP4、DT_FP4E2M1、DT_FP4E1M2、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_INT4、DT_UINT4、DT_HF4、DT_HF8等仅用于存储的低精度类型。写入值类型须与元素类型一致或可由前端按该元素类型构造。 |
| offset | 输入 | 写入位置，以元素为单位的线性偏移。支持整型常量或运行时整型标量表达式（包括循环变量）；取值范围为0 ≤ offset < 总元素数，越界行为不确定。也可使用`container[i] = value`或`container[i, j, ...] = value`按坐标写入，编译器会将下标转换为线性偏移；下标个数须与容器的rank一致。 |
| value | 输入 | 要写入的标量值。整型或浮点型常量，或运行时整型或浮点型标量表达式，类型须与container元素类型兼容。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

### Tile场景

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def getval_setval_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_a_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        pl.load(tile_a, a, [0, 0])
        value = pl.getval(tile_a, 0)
        pl.setval(tile_a, 1, value)
        pl.store(a, tile_a, [0, 0])
```

### Tensor场景

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tensor_getval_setval_kernel(
    scale_tensor: pl.Tensor[[2], pl.DT_FP32],
):
    scale = pl.getval(scale_tensor, 0)
    pl.setval(scale_tensor, 1, scale)
```
