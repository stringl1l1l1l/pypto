# pypto_pro.language.range

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

生成Kernel中for循环的迭代区间。迭代区间由起始值、终止值和步长确定，不包含终止值。循环变量为运行时整数标量，可用于Tile索引、数据偏移计算和标量运算。

## 函数原型

```python
pypto_pro.language.range(
    stop: Union[int, Scalar],
) -> RangeIterator

pypto_pro.language.range(
    start: Union[int, Scalar],
    stop: Union[int, Scalar],
    step: Union[int, Scalar] = 1,
) -> RangeIterator
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| start | 输入 | 起始值，支持整型常量、运行时整型标量或整型标量表达式，单参数形式下省略并取0，取值范围见[约束说明](#约束说明)。 |
| stop | 输入 | 终止值（不包含），支持整型常量、运行时整型标量或整型标量表达式，取值范围见[约束说明](#约束说明)。 |
| step | 输入 | 步长，支持整型常量、运行时整型标量或整型标量表达式，取值范围见[约束说明](#约束说明)。 |

## 约束说明

- 只支持1～3个位置参数，不支持关键字传参。
- start、stop、step参数取值范围要求如下（仅编译期常量能校验拦截，运行时标量或表达式的取值需用户自行保证）
    - start、stop、step必须为整型标量。
    - start、stop的取值范围与所在执行域相关
        - VF域（pypto_pro.language.vector_function修饰的函数体）内：start、stop必须为非负数，且不超过uint16，即取值范围为[0, 65535]。
        - VF域外：start、stop取值范围为int64，即[-9223372036854775808, 9223372036854775807]。
    - step必须为正整数，且不超过所在执行域的上界
        - VF域内：step取值范围为[1, 65535]。
        - VF域外：step取值范围为[1, 9223372036854775807]。
    - 循环变量在最后一次自增（i += step）后的取值不能超过其所在执行域的上界（VF域内为65535，VF域外为int64上界），否则会发生回绕或溢出，导致循环结果错误。
- 支持嵌套循环，也支持在for循环中使用break和continue。break只退出其所在的最内层循环，continue跳过当前迭代。
- pypto_pro.language.range可在JIT函数的公共函数体、pypto_pro.language.section_vector或pypto_pro.language.section_cube内部使用。循环体中的具体操作仍需满足相应执行域的使用约束。

## 返回值说明

返回一个用于for循环的迭代器。

## 调用示例

### 双层循环分块

```python
import pypto_pro.language as pl

# 本示例要求M和N分别是TILE_M和TILE_N的整数倍。
# Vector Kernel开启auto_mutex，同步由make_tile_group自动管理。
TILE_M = 64
TILE_N = 64


@pl.jit(auto_mutex=True)
def for_add_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    M = x.shape[0]
    N = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, M // TILE_M, 1):
            for j in pl.range(0, N // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
```

### 单参数形式

```python
# 等价于range(0, 10, 1)。
for i in pl.range(10):
    ...
```
