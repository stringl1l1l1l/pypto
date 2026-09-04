# pypto_pro.language.getval

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

读取Tile或Tensor指定位置的单个元素值，返回标量。根据第一个参数的类型（Tile或Tensor）自动分发到对应的后端实现。推荐使用下标语法糖container[i, j]替代pypto_pro.language.getval(container, offset)，下标语法自动将多维坐标线性化为偏移量，语义更清晰；线性偏移API适用于需要直接计算线性地址的场景（如跨rank共享helper）。

## 函数原型

```python
# 方式一：下标语法糖（推荐）
value = container[i]              # 1D 容器
value = container[i, j]           # 多维容器（索引数 = rank）

# 方式二：线性偏移 API
value = pypto_pro.language.getval(container, offset)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| container | 输入 | 目标Tile或Tensor，从中读取单个元素。Tile必须位于UB；Tensor为GM Tensor。支持可参与标量表达式的整型或浮点类型，不支持DT_FP4、DT_FP8E4M3FN、DT_FP8E5M2、DT_INT4、DT_UINT4、DT_HF4、DT_HF8等仅用于存储的低精度类型。 |
| i, j, ... | 输入 | 下标语法糖的多维索引，均为整数，索引数必须等于容器rank，1D容器可用单索引container[i]。多维索引自动线性化为`i * (N1*N2*...) + j * (N2*...) + ...`。 |
| offset | 输入 | 线性偏移API的读取位置，线性元素偏移。整型常量或运行时整型标量表达式（支持循环变量），取值范围0 ≤ offset < 总元素数，越界行为不确定。 |

## 约束说明

无。

## 返回值说明

返回与container元素类型一致的标量。

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
        value = tile_a[0, 0]      # 读 Tile[0,0] 元素
        tile_a[0, 1] = value      # 写到 Tile[0,1] 位置
        pl.store(a, tile_a, [0, 0])
```

### Tensor场景

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tensor_getval_setval_kernel(
    scale_tensor: pl.Tensor[[2], pl.DT_FP32],
):
    scale = scale_tensor[0]
    scale_tensor[1] = scale
```
