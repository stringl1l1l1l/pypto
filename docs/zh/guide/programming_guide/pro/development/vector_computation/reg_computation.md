# Reg计算

Reg计算直接使用SIMD Register File保存矢量数据和中间结果。PyPTO Pro通过@pypto_pro.language.vector_function定义VF函数，并在函数内使用[vf.* API](../../../../../api/pro_api/SIMD-API/reg_computation/index.md)表达寄存器加载、计算和存储。

> [!NOTE]说明
> Reg计算依赖VF Register File，使用前请确认对应VF API的支持范围。

## Reg计算的适用场景

Tile计算以UB Tile为数据载体。多个矢量操作串联时，中间结果通常需要写回UB，再由下一条指令读取。计算链较长时，反复访问UB会增加读写带宽压力和Bank冲突概率。

Reg计算将一段连续计算保留在寄存器中，仅在计算链入口和出口与UB交互：

| 维度 | Tile计算 | Reg计算 |
|:---|:---|:---|
| 数据载体 | UB中的Tile | Register File中的RegTensor / MaskReg |
| 中间结果 | 通常写回UB | 可由后续vf.*操作直接消费 |
| PyPTO Pro接口 | pypto_pro.language.add、pypto_pro.language.sub、pypto_pro.language.sum等 | vf.add、vf.sub、vf.reduce_*等 |
| 适用场景 | 通用矢量计算、快速实现 | 连续计算链、需要降低UB往返开销的高性能场景 |

## 硬件组成

Vector侧参与Reg计算的硬件单元包括：

- **Reg矢量执行单元**：从Register File读取操作数并将计算结果写回寄存器。
- **DMA单元**：在UB与Register File之间搬运数据。
- **Aux Scalar**：完成VF域内的地址、循环等标量计算。

**图1 SIMD Reg矢量执行关系**

![SIMD Reg矢量执行单元与Register File、UB的关系](../../../../figures/pro/register_execution_unit.jpg)

## 内存层级

Register File位于UB之上，不能直接从GM加载或直接写回GM。完整数据路径是：

```text
GM → UB → Register File → UB → GM
```

**图2 Reg计算内存层级**

![Register File、UB和GM的层级关系](../../../../figures/pro/register_memory_hierarchy.jpg)

PyPTO Pro中各阶段的接口对应关系如下：

| 数据路径 | PyPTO Pro表达 |
|:---|:---|
| GM → UB | pypto_pro.language.load / pypto_pro.language.load_tile |
| UB → Register File | vf.load / vf.load_align / vf.load_unalign等 |
| Register File内计算 | vf.add、vf.mul、vf.reduce_sum等 |
| Register File → UB | vf.store / vf.store_align / vf.store_unalign等 |
| UB → GM | pypto_pro.language.store / pypto_pro.language.store_tile |

## 编程模型

Regbase在Tile计算的“数据搬入 → 计算 → 数据搬出”基础上，将矢量计算阶段细分为“Load → Compute → Store”。

**图3 Regbase编程模型总体结构**

![GM、UB、Register File之间的Regbase编程流程](../../../../figures/pro/regbase_programming_model_overview.jpg)

### VF函数与执行域

使用@pypto_pro.language.vector_function声明VF函数。函数体隐式处于VF执行域，使用vf.*操作加载、计算和存储寄存器；Tile参数的类型由调用点推导。

vf.*操作只能在VF函数内使用，放在pypto_pro.language.section_vector()内仍需要通过VF函数调用。
VF函数可以调用其他VF函数，并使用标量表达式、pypto_pro.language.range循环以及标量pypto_pro.language.min、pypto_pro.language.max、pypto_pro.language.const。
其他pypto_pro.language.*调用（包括Tile操作、同步操作和pypto_pro.language.section_vector()/pypto_pro.language.section_cube()）应放在VF函数外，
需要的结果通过参数传入。TileGroup的next()/current()/previous()同样如此：游标推进会产生标量运算，
不能进入VF执行域，应在调用侧选好Tile后作为参数传入。pypto_pro.language.DT_*和枚举常量仍可在VF函数内使用。
违反执行域限制时，前端解析器会在对应调用处报错。

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


@pl.vector_function
def add_vf(src_a, src_b, dst):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.add(reg_a, reg_b, preg)
    vf.store_align(dst, reg_out, preg)
```

外层@pypto_pro.language.jit Kernel负责GM与UB之间的搬运以及跨Pipe同步，并在pypto_pro.language.section_vector()中调用VF函数：

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[1, 64], pl.DT_FP32],
    b: pl.Tensor[[1, 64], pl.DT_FP32],
    out: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec)
    a_group = pl.make_tile_group(type=tt, addrs=0x000, mutex_ids=[0])
    b_group = pl.make_tile_group(type=tt, addrs=0x100, mutex_ids=[1])
    out_group = pl.make_tile_group(type=tt, addrs=0x200, mutex_ids=[2])

    with pl.section_vector():
        tile_a = a_group.current()
        tile_b = b_group.current()
        tile_out = out_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        add_vf(tile_a, tile_b, tile_out)
        pl.store(out, tile_out, [0, 0])
```

完整寄存器生命周期说明参见[vf.reg_tensor](../../../../../api/pro_api/SIMD-API/reg_computation/basic_data_structures/reg_tensor.md)。

### VF函数中的Tile指针偏移

VF函数接收的Tile参数可以使用tile + offset进行线性元素偏移，偏移后的表达式可传给vf.load_align、vf.store_align等访存接口。例如，下面的VF函数按行读取源Tile，并将结果连续写入目标Tile：

```python
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

@pl.vector_function
def copy_rows(dst_tile, src_tile, row_count, col_count, src_stride):
    preg = vf.update_mask(col_count, dtype=pl.DT_FP16)
    for row in pl.range(row_count):
        vreg = vf.load_align(src_tile, row * src_stride)
        vf.store_align(dst_tile + row * col_count, vreg, preg)
```

offset的单位是元素，可以是整型常量或运行时整型Scalar。tile + offset只形成偏移后的指针表达式，不会创建新的Tile，也不携带shape或valid_shape信息。

VF函数内不支持tile[row_start:row_stop, col_start:col_stop]切片。如果需要先选取二维区域，应在pypto_pro.language.section_vector()中创建子Tile，再将其传给VF函数：

```python
import pypto_pro.language as pl

with pl.section_vector():
    src_tile = src_group.next()
    dst_tile = dst_group.next()
    pl.load(src_tile, src, [0, 0])
    copy_rows(dst_tile, src_tile[1:4, 16:48], 3, 32, 64)
    pl.store(dst, dst_tile, [0, 0])
```

## 同步与依赖

- GM↔UB搬运与Vector/VF计算之间的跨Pipe依赖，由TileGroup + auto_mutex=True自动管理，或使用pypto_pro.language.system.sync_src / pypto_pro.language.system.sync_dst手动管理。
- VF函数内存在UB写后读、写后写等局部依赖时，按接口要求使用vf.mem_bar指定对应模式。
- Register File中存在直接数据依赖的vf.*表达式应保持清晰的数据流关系，避免在未初始化寄存器上执行计算。

## 使用建议

Reg计算和[Tile计算](tile_computation.md)分别侧重性能与易用性，可根据算子的开发需求进行选择：

- **Reg计算（VF计算）**：中间结果可以保留在Vector Register File中，减少UB读写，性能更优；使用前需要确认目标设备和所需VF API均受支持，并遵守寄存器、数据类型和接口约束。
- **Tile计算**：接口和数据流更直观，开发、调试及维护更方便；中间结果通常需要通过UB读写，性能相对较低。
