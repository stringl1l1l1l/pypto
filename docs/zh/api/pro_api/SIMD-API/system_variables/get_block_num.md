# pypto_pro.language.get_block_num

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

获取本次实际启动的逻辑Block数量，用于多核控制和数据偏移计算。

## 函数原型

```python
pypto_pro.language.get_block_num() -> int
```

## 参数说明

无。

## 约束说明

无。

## 返回值说明

返回DT_INT64类型的标量值，表示限核后实际启动的Block数，可用于Kernel内的整数计算和数据索引。
JIT每次启动通过C++启动器查询实际Stream的有效资源限制，按Kernel执行域和配对比例限制Host请求的`block_dim`。
因此返回值可能小于Host请求值；数据切分应使用本接口返回值作为循环步长，避免遗漏任务。

仅启动Cube（AIC）或仅启动Vector（AIV）时，该值等于执行域逻辑核数。
在AIC:AIV为1:2的混合Kernel中，该值表示逻辑Block数；AIC逻辑核数为`get_block_num()`，
AIV逻辑核数为`get_block_num() * get_subblock_num()`。

## 调用示例

### 按实际Block数循环切分

用Kernel[None, NUM_CORES]请求最多2个逻辑Block，每个AIV以get_block_num()返回值为步长跨步处理64行Tile。用[pypto_pro.language.printf](../../Utils-API/debugging/printf.md)打印实际启动的Block数：

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

### 限核启动下的返回值

启动Block数是请求的上界，实际启动数可能更小。复用上例Kernel与`__main__`中的x/y，仅把启动Block数改为1：

```python
    z = torch.zeros([128, 128], device=device, dtype=torch.float16)
    multicore_add_kernel[None, 1](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y, rtol=1e-2, atol=1e-2)  # 仍覆盖全部128行
```

回显：

```text
=> Vec 0
block_idx = 0, block_num = 1
```
