# pypto_pro.language.system.dcci

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

对指定地址对应的数据缓存执行清理并失效（Data Cache Clean and Invalidate，DCCI）。典型用途是在跨核或跨流水共享数据时，清除当前执行单元可能持有的旧缓存副本，使后续访问能够观察到已发布的数据。

dcci只处理缓存状态，不等价于流水同步、跨核事件或内存屏障。调用者仍须保证生产者写入已经完成，并使用与通信协议匹配的同步接口建立先后关系。

## 函数原型

```python
pypto_pro.language.system.dcci(
    target: Union[Tensor, Tile],
    offset: Optional[Union[int, Sequence[int], Expr]] = None,
    *,
    cache_line: CacheLine = pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE,
    dst: DcciDst = pypto_pro.language.DcciDst.AUTO,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | GM Tensor变量，或已分配在UB中的Tile。 |
| offset | 输入 | 可选，元素偏移，单位为target.dtype元素。GM Tensor支持各维偏移列表/元组，也支持整型常量或运行时整型标量表达式表示的线性偏移；列表/元组长度须与Tensor维数一致，框架按Tensor stride换算线性偏移。UB Tile仅支持整型常量或运行时整型标量表达式表示的线性偏移。缺省时为0，即使用目标起始地址；偏移不得使有效地址越出目标已分配范围。 |
| cache_line | 输入 | 可选，编译期[pypto_pro.language.CacheLine](../basic_data_structures/CacheLine.md)枚举值，默认pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE。缓存行为64字节；SINGLE_CACHE_LINE的地址无需由用户向下对齐，硬件操作包含该地址的缓存行。若数据跨越多个缓存行，须逐行调用或使用ENTIRE_DATA_CACHE。 |
| dst | 输入 | 可选，编译期[pypto_pro.language.DcciDst](../basic_data_structures/DcciDst.md)枚举值，默认pypto_pro.language.DcciDst.AUTO。各枚举值的含义和适用目标参见DcciDst。 |

## 约束说明

- cache_line和dst必须在编译期确定，不能由运行时Scalar或Tensor动态选择。
- SINGLE_CACHE_LINE只覆盖一个64字节缓存行。处理地址区间[addr, addr + bytes)时，调用次数至少为该区间覆盖的缓存行数，不能只对首地址调用一次。
- ENTIRE_DATA_CACHE作用于整个数据缓存，offset不会缩小其作用范围；该模式开销大于单缓存行操作。
- DCCI不是同步原语。生产者通过MTE3等流水写出数据后，必须先同步到S流水再执行DCCI或发布标志；消费者也必须先完成相应跨核等待，再执行缓存处理和数据读取。具体事件号和同步模式由上层通信协议决定。
- 频繁对整个缓存执行DCCI会造成明显性能损失；已知共享数据范围时应优先按64字节缓存行处理。

## 返回值说明

无。

## 调用示例

### 单缓存行失效（GM Tensor）

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit()
def dcci_gm_kernel(
    inp: pl.Tensor[[16, 16], pl.DT_FP32],
    out: pl.Tensor[[16, 16], pl.DT_FP32],
):
    # 对 inp 起始元素所在地址的64字节缓存行执行清理并失效
    pl.system.dcci(inp, [0, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    inp = torch.ones([16, 16], device=device, dtype=torch.float32)
    out = torch.ones([16, 16], device=device, dtype=torch.float32)

    dcci_gm_kernel(inp, out)
    torch.npu.synchronize()

    # DCCI只处理缓存状态，不修改数据
    assert torch.allclose(out.cpu(), inp.cpu())
    print("dcci done")
```

### 全缓存失效

```python
# 对整个数据缓存执行DCCI，开销大于单缓存行操作，offset不会缩小其作用范围
pl.system.dcci(inp, cache_line=pl.CacheLine.ENTIRE_DATA_CACHE)
```

### DCCI搭配同步使用

对UB Tile执行DCCI时，须先用sync_src/sync_dst建立流水先后关系（DCCI不是同步原语）：

```python
@pl.jit(auto_mutex=True)
def dcci_ub_kernel(
    inp: pl.Tensor[[16, 16], pl.DT_FP32],
    out: pl.Tensor[[16, 16], pl.DT_FP32],
):
    tt = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    t = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(t, inp, [0, 0])
        # load（MTE2）完成后才能执行 dcci（S）
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.dcci(t, cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        # dcci（S）完成后再 store（MTE3），否则store可能读到失效前的旧数据
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t, [0, 0])
```
