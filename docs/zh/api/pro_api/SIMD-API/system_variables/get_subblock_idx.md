# pypto_pro.language.get_subblock_idx

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

获取当前逻辑AI Core内AIC或AIV的subblock索引。

## 函数原型

```python
pypto_pro.language.get_subblock_idx() -> int
```

## 参数说明

无。

## 约束说明

无。

## 返回值说明

返回DT_INT64类型的subblock索引，可用于Kernel内的整数计算和数据索引。取值范围为
[0, get_subblock_num())；在AIC与AIV比例为1:2的混合Kernel中，同一逻辑Block对应的两个AIV分别返回0和1。

## 调用示例

### 纯Vector Kernel中读取子核号

纯Vector Kernel中每个Block即一个AIV（get_subblock_num()返回1），get_subblock_idx()恒为0：

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def subblock_add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_sum = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        sub_idx = pl.get_subblock_idx()
        pl.printf("subblock_idx = %d, subblock_num = %d\n", sub_idx, pl.get_subblock_num())
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_sum = tile_sum.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_sum, cur_x, cur_y)
        pl.store(out, cur_sum, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.randn([64, 64], device=device, dtype=torch.float32)
    y = torch.randn([64, 64], device=device, dtype=torch.float32)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)

    subblock_add_kernel(x, y, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + y, rtol=1e-5, atol=1e-5)
    print(f"max diff = {(out - (x + y)).abs().max().item()}")
```

回显：

```text
=> Vec 0
subblock_idx = 0, subblock_num = 1
```

### 混合Kernel中的子核号与条件执行

混合Kernel的Vector段中，get_block_idx()返回全局AIV逻辑索引用于数据分片，get_subblock_idx()用于区分同一逻辑Block内的两个AIV。根据子核号可做条件执行，例如只让sub-core 0写结果：

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit()
def subblock_cond_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_cube():
        pass
    with pl.section_vector():
        block_idx = pl.get_block_idx()      # 全局AIV逻辑索引：0~3
        sub_idx = pl.get_subblock_idx()     # 块内AIV编号：0或1
        pl.printf("block_idx = %d, subblock_idx = %d\n", block_idx, sub_idx)
        # 条件执行：只让每个逻辑Block的sub-core 0写结果
        if sub_idx == 0:
            pl.printf("sub-core 0 of core %d writes output\n", block_idx // pl.get_subblock_num())
            pl.setval(out, 0, 1)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    out = torch.zeros(1, device=device, dtype=torch.int32)

    subblock_cond_kernel[None, 2](out)
    torch.npu.synchronize()
```

回显（启动2个逻辑Block；`=> Cube`条目为Cube段，本例无打印）：

```text
=> Cube 0

=> Vec 0
block_idx = 0, subblock_idx = 0
sub-core 0 of core 0 writes output

=> Vec 1
block_idx = 1, subblock_idx = 1

=> Cube 1

=> Vec 2
block_idx = 2, subblock_idx = 0
sub-core 0 of core 1 writes output

=> Vec 3
block_idx = 3, subblock_idx = 1
```
