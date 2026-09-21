# pypto_pro.language.get_block_idx

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

获取当前实际执行核在所属执行域中的全局索引，用于多核控制和数据偏移计算。

## 函数原型

```python
pypto_pro.language.get_block_idx() -> int
```

## 参数说明

无。

## 约束说明

无。

## 返回值说明

返回DT_INT64类型的当前逻辑核索引，可用于Kernel内的整数计算和数据索引。取值范围与Kernel的执行域有关：

- 仅启动Cube（AIC）或仅启动Vector（AIV）时，范围为[0, get_block_num())。
- 同时启动AIC与AIV时，AIC侧范围为[0, get_block_num())；AIV侧范围为[0, get_subblock_num() × get_block_num())。当前1:2配置下，AIV侧范围为[0, 2 × get_block_num())。

在Vector段中，该接口返回全局AIV逻辑索引，编号方式为：物理核号 × get_subblock_num() + 子核号，可直接用于数据分片和偏移计算。混合Kernel在Vector段做跨步切分时，工作单元总数为get_block_num() × get_subblock_num()。

## 调用示例

### 多核数据分片（纯Vector Kernel）

通过Kernel[None, NUM_CORES]形式实际启动2个AIV，每个AIV用get_block_idx()获取全局逻辑索引并处理64行逐元素加法，用[pypto_pro.language.printf](../../Utils-API/debugging/printf.md)打印获取到的索引值：

```python
import os
import pypto_pro.language as pl
import torch

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def multicore_add_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        vidx = pl.get_block_idx()              # 当前AIV的全局逻辑索引
        num_blocks = pl.get_block_num()        # 实际启动的Block数
        pl.printf("block_idx = %d, block_num = %d\n", vidx, num_blocks)
        for tile_idx in pl.range(vidx, 2, num_blocks):
            offset = tile_idx * 64
            cur_a = tile_a.current()
            cur_b = tile_b.current()
            cur_c = tile_c.current()
            pl.load(cur_a, x, [offset, 0])
            pl.load(cur_b, y, [offset, 0])
            pl.add(cur_c, cur_a, cur_b)
            pl.store(z, cur_c, [offset, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.rand([128, 128], device=device, dtype=torch.float16)
    y = torch.rand([128, 128], device=device, dtype=torch.float16)
    z = torch.zeros([128, 128], device=device, dtype=torch.float16)

    multicore_add_kernel[None, NUM_CORES](x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y, rtol=1e-2, atol=1e-2)
    print(f"max diff = {(z - (x + y)).abs().max().item()}")
```

回显（`=> Vec`后为核号，多核间输出顺序不固定）：

```text
=> Vec 0
block_idx = 0, block_num = 2

=> Vec 1
block_idx = 1, block_num = 2
```

### 混合Kernel中AIC与AIV的返回值

混合Kernel（AIC:AIV=1:2）中，Cube段返回物理核号[0, get_block_num())，Vector段返回全局AIV逻辑编号[0, get_block_num() × get_subblock_num())，编号方式为：物理核号 × get_subblock_num() + 子核号：

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit()
def block_idx_mix_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_cube():
        aic_idx = pl.get_block_idx()
        pl.printf("[cube] block_idx = %d\n", aic_idx)
        pl.setval(out, 0, 1)
    with pl.section_vector():
        aiv_idx = pl.get_block_idx()
        pl.printf("[vector] block_idx = %d\n", aiv_idx)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    out = torch.zeros(1, device=device, dtype=torch.int32)

    block_idx_mix_kernel[None, 2](out)
    torch.npu.synchronize()
```

回显（实际启动2个AIC和4个AIV：AIC侧返回0/1，AIV侧返回0~3）：

```text
=> Cube 0
[cube] block_idx = 0

=> Vec 0
[vector] block_idx = 0

=> Vec 1
[vector] block_idx = 1

=> Cube 1
[cube] block_idx = 1

=> Vec 2
[vector] block_idx = 2

=> Vec 3
[vector] block_idx = 3
```
