# pypto_pro.language.Tensor

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

GM中多维张量类型标注，主要用于：

- Kernel函数中GM张量参数的类型声明。
- 配合[pypto_pro.language.load](../memory_data_movement/load.md)/[pypto_pro.language.store](../memory_data_movement/store.md)完成GM与L1 Buffer/UB/L0C Buffer之间的数据搬运。
- 通过等号赋值创建别名，与原张量共享同一段GM内存，不产生数据拷贝。支持链式别名；别名一经创建即固定指向，之后将原变量名重新绑定到其他张量，不会改变已有别名的指向。

## 函数原型

```python
pypto_pro.language.Tensor.__init__(
    self,
    shape: Sequence[int | _ShapePolicy | EllipsisType] | None = None,
    dtype: DataType | None = None,
    expr: Expr | None = None,
    layout: TensorLayout | None = None,
    memref: MemRef | None = None,
    _annotation_only: bool = False,
    direction: str | None = None,
)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| shape | 输入 | 各维大小列表。<br>- 表现形式如下：<br>&nbsp;&nbsp;- 固定维度：正整数表示，如[64, 128]，调用时对应维度须等于该整数。<br>&nbsp;&nbsp;- 动态维度：pypto_pro.language.DYNAMIC表示，调用时读取实际维度，维度值不参与编译缓存键，不同取值复用同一编译变体。<br>&nbsp;&nbsp;- 编译期特化维度：pypto_pro.language.STATIC表示，调用时读取实际维度，维度值固化到当前编译变体，取值变化时生成新的编译变体。<br>&nbsp;&nbsp;- 末尾...：展开剩余维度，各维均按pypto_pro.language.STATIC处理。<br>&nbsp;&nbsp;- 不同策略可混用，如[64, pl.DYNAMIC, pl.STATIC]。<br>- Kernel内可通过tensor.shape[i]读取对应维度，返回值类型为DT_INT64。 |
| dtype | 输入 | 元素数据类型，[pypto_pro.language.DataType](DataType.md)枚举值。 |
| expr | 输入 | 可选，框架内部使用的IR表达式，用于构造与底层Expr关联的运行时Tensor对象。不能与shape或dtype同时配置，用户无需设置。 |
| layout | 输入 | 可选，内存布局，[pypto_pro.language.TensorLayout](TensorLayout.md)枚举值。<br>- 支持ND和NZ，不指定时按ND处理。<br>- 配置为NZ时只声明布局，不执行ND→NZ转换。 |
| memref | 输入 | 可选，显式内存引用，pypto_pro.language.MemRef实例。三参数形式中第三项为MemRef实例时按memref解析；需要同时指定layout和memref时使用四参数形式。 |
| _annotation_only | 输入 | 框架内部参数，用于标识当前Tensor是否仅作为类型标注使用，用户无需设置。 |
| direction | 输入 | 仅用于类型标注。可选，Tensor参数在Profiling元数据中的方向，取值为pypto_pro.language.Input或pypto_pro.language.Output，省略时默认为pypto_pro.language.Input。方括号形式中可放在dtype后的任一可选位置；调用式中仅支持作为第三个位置参数传入。 |

## 约束说明

- 每个Tensor类型标注最多声明一个direction，取值只能为`pl.Input`或`pl.Output`。
- 搬运约束详见[pypto_pro.language.load](../memory_data_movement/load.md)和[pypto_pro.language.store](../memory_data_movement/store.md)。

## 返回值说明

无。

## 调用示例

### 类型声明

```python
import pypto_pro.language as pl

# 固定整数维度
x: pl.Tensor[[64, 128], pl.DT_FP16]

# 带布局的tensor
y: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]

# 高维NZ：最后两轴64/128为M/N，前两轴为batch
y_4d: pl.Tensor[[2, 4, 64, 128], pl.DT_FP16, pl.NZ]

# A矩阵的E8M0分组缩放因子：逻辑shape为[M,G]=[64,4]，GM物理shape为[M,G/2,2]=[64,2,2]
scale_a: pl.Tensor[[64, 2, 2], pl.DT_FP8E8M0]

# 动态维度声明（仅用于类型标注）
dynamic_tensor: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]
```

### Tensor参数方向标注

Kernel形参的Tensor类型标注支持使用`pypto_pro.language.Input`和`pypto_pro.language.Output`两个枚举值声明该参数在Profiling元数据中的方向：

```python
import pypto_pro.language as pl

# 方括号形式。
input_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.Input]
output_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.Output]

# direction可与layout组合，推荐放在标注末尾。
nz_output: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ, pl.Output]

# 调用式的第三个参数为direction。
call_style_output: pl.Tensor([64, 128], pl.DT_FP16, pl.Output)
```

`pl.Input`表示输入，`pl.Output`表示输出。direction可省略，未标注时默认使用`pl.Input`。该标注用于设置Profiling结果中的Tensor输入、输出信息，不改变Kernel的读写语义。

### Tensor别名

```python
import pypto_pro.language as pl

# 一级别名和链式别名均指向首次传入的input_tensor
original_input_alias = input_tensor
original_input_alias_chain = original_input_alias

# 别名可作为Tensor操作数
pl.load(input_tile, original_input_alias_chain, [0, 0])

# 重新绑定原变量不改变已有别名的指向
input_tensor = replacement_tensor
pl.load(input_tile, original_input_alias, [0, 0])   # 仍从首次传入的input_tensor读取
pl.load(replacement_tile, input_tensor, [0, 0])     # 从replacement_tensor读取
```

### DYNAMIC动态维度

```python
import os

import pypto_pro.language as pl
import torch


@pl.jit(auto_mutex=True)
def dynamic_tensor_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_out, tile_a, tile_b)
        pl.store(out, tile_out, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    a = torch.randn([64, 64], device=device, dtype=torch.float32)
    b = torch.randn([64, 64], device=device, dtype=torch.float32)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)

    dynamic_tensor_kernel(a, b, out)
    torch.npu.synchronize()

    ref = a + b
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### STATIC编译期特化维度

```python
import os

import pypto_pro.language as pl
import torch

TILE_M = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_static(
    x: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    y: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    z: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    # shape为STATIC，首次调用[256, 256]生成一个编译变体
    x = torch.randn([256, 256], device=device, dtype=torch.float16)
    y = torch.randn([256, 256], device=device, dtype=torch.float16)
    z = torch.zeros([256, 256], device=device, dtype=torch.float16)

    add_static[None, 8](x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y, rtol=1e-3, atol=1e-3)
    print(f"[256, 256] max diff = {(z - (x + y)).abs().max().item()}")
```
