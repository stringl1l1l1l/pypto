# Softmax算子快速入门（SIMT）

## 任务与目标

本节介绍如何参考SIMD版的单Kernel流程，使用PyPTO Pro SIMT编程范式实现Softmax，并通过测试用例验证结果。您将了解如何将数据搬入片上Tile、在SIMT函数内完成归约与逐元素计算，以及将结果搬回全局内存。

本示例固定处理Shape为[1, 256]、数据类型为FP32的Tensor。一个Kernel启动256个SIMT线程，每个线程负责Tile中的一个元素；示例要求输入元素为有限数值。

## 算子设计规格

**表1** Softmax算子设计规格

| name            | shape    | data type | format |
| --------------- | -------- | --------- | ------ |
| **inputs（输入）**  | (1, 256) | float32   | ND     |
| **outputs（输出）** | (1, 256) | float32   | ND     |

* 数学表达式

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i-M)}{\sum_j \exp(z_j-M)}$

* 使用的主要接口

  Tile定义与数据搬运接口：pypto_pro.language.TileType、pypto_pro.language.make_tile_group、pypto_pro.language.load、pypto_pro.language.store

  线程索引与块内同步接口：pypto_pro.language.simt.linear_thread_idx、pypto_pro.language.simt.syncthreads

  指数与原子归约接口：pypto_pro.language.simt.exp、pypto_pro.language.simt.atomic_max、pypto_pro.language.simt.atomic_add

## 导入PyPTO Pro模块

在开始实现Softmax算子之前，首先需要导入PyPTO Pro、PyTorch和torch_npu模块。PyPTO Pro模块用于定义SIMT函数和编译Kernel，PyTorch和torch_npu模块用于构造输入数据、执行NPU计算并进行结果验证。

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu
```

本示例输入最后一维的长度为256，因此设置SIMT线程数为256：

```python
THREADS = 256
```

## 核心代码逻辑

1. 实现Tile上的SIMT计算函数。

   与SIMD版先将数据搬入Tile再计算的流程一致，这里由一个SIMT函数完成最大值归约、指数求和及归一化。256个线程各处理一个元素，使用同一块内的stats_tile保存最大值与指数和。

   ```python
   @pl.vector_function(mode="simt", max_threads=THREADS)
   def softmax_tile(src_tile, out_tile, stats_tile):
       tid = pl.simt.linear_thread_idx()

       if tid == 0:
           stats_tile[0, 0] = src_tile[0, 0]
           stats_tile[0, 1] = 0.0
       pl.simt.syncthreads()

       value = src_tile[0, tid]
       pl.simt.atomic_max(stats_tile[0, 0], value)
       pl.simt.syncthreads()

       exp_value = pl.simt.exp(value - stats_tile[0, 0])
       pl.simt.atomic_add(stats_tile[0, 1], exp_value)
       pl.simt.syncthreads()

       out_tile[0, tid] = exp_value / stats_tile[0, 1]
   ```

   第一个线程初始化共享归约状态；每次归约结束后，所有线程通过syncthreads()等待结果可见。三个屏障均由所有线程无条件到达。最大值从输入首元素初始化，避免额外的FP32最小值常量。

2. 实现Softmax Kernel函数。

   与SIMD版相同，Kernel先分配片上Tile Group，再在Vector段完成数据搬入、计算和数据搬出。输入与输出Tile各占1024字节；归约状态Tile占32字节，其中前两个FP32元素分别保存最大值与指数和。三个Tile Group使用互不重叠的片上地址和不同的Mutex ID。

   ```python
   @pl.jit(arch="3510", auto_mutex=True)
   def softmax_simt_kernel(
       src: pl.Tensor[[1, THREADS], pl.DT_FP32],
       dst: pl.Tensor[[1, THREADS], pl.DT_FP32],
   ):
       tile_type = pl.TileType(
           shape=[1, THREADS],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
       )
       stats_type = pl.TileType(
           shape=[1, 8],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
       )
       src_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
       out_group = pl.make_tile_group(type=tile_type, addrs=0x0400, mutex_ids=[1])
       stats_group = pl.make_tile_group(type=stats_type, addrs=0x0800, mutex_ids=[2])

       with pl.section_vector():
           src_tile = src_group.current()
           out_tile = out_group.current()
           stats_tile = stats_group.current()

           pl.load(src_tile, src, [0, 0])
           softmax_tile[THREADS](src_tile, out_tile, stats_tile)
           pl.store(dst, out_tile, [0, 0])
   ```

   auto_mutex=True结合Tile Group的Mutex ID自动处理搬入、SIMT计算与搬出之间的流水依赖，不需要手动调用sync_src/sync_dst。SIMT函数内部的syncthreads()仍负责线程间的数据依赖，不能省略。

## 测试用例

使用PyTorch Tensor构造普通输入和较大正负偏移输入，将单次Kernel的结果与PyTorch Softmax对比。由于本示例只处理一行，显式使用kernel[None, 1]启动一个核心；softmax_tile[THREADS]在单个aiv中启动256个SIMT线程。

```python
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"

torch.npu.set_device(device)

torch.manual_seed(0)
for offset in (0.0, 100.0, -100.0):
    src = torch.randn(1, THREADS, dtype=torch.float32, device=device) + offset
    dst = torch.empty_like(src)

    softmax_simt_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    golden = torch.softmax(src, dim=-1)
    torch.testing.assert_close(dst, golden, rtol=1e-4, atol=1e-5)

print("SIMT Softmax kernel passed!")
```

归约状态在每次SIMT函数调用时由线程0初始化，不需要Host侧的中间Tensor。FP32原子加的累加顺序不固定，因此验证使用容差而非逐位比较。

## 编译与执行

在已安装PyPTO Pro的环境中运行：

```bash
# 配置CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备ID
export TILE_FWK_DEVICE_ID=0

# 执行脚本
python3 softmax_simt.py
```

程序执行成功后，显示以下信息：

```text
SIMT Softmax kernel passed!
```
