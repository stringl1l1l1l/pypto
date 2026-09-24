# Softmax算子快速入门（SIMD）

## 任务与目标

本节将详细介绍如何使用PyPTO Pro SIMD编程范式实现一个简单的Softmax算子，并通过测试用例验证其正确性。通过本节的学习，您将了解如何使用PyPTO Pro SIMD相关API完成Tile定义、片上存储空间分配、数据搬运、归约计算和逐元素计算。

本示例固定处理Shape为[8, 256]、数据类型为FP32的Tensor，显式定义计算过程中使用的Tile及其片上存储空间，并通过SIMD接口沿最后一维分别对每行执行Softmax，其中第一维设置为8，以满足FP32行归约结果Tile的存储对齐要求。

## 算子设计规格

**表1** Softmax算子设计规格

| name            | shape    | data type | format |
| --------------- | -------- | --------- | ------ |
| **inputs（输入）**  | (8, 256) | float32   | ND     |
| **outputs（输出）** | (8, 256) | float32   | ND     |

* 数学表达式

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i-M)}{\sum_j \exp(z_j-M)}$

* 使用的主要接口

  Tile定义与分配接口：pypto_pro.language.TileType、pypto_pro.language.make_tile_group

  数据搬运接口：pypto_pro.language.load、pypto_pro.language.store

  基础计算接口：pypto_pro.language.maximum、pypto_pro.language.expand_sub、pypto_pro.language.exp、pypto_pro.language.sum、pypto_pro.language.expand_div

## 导入PyPTO Pro模块

在开始实现Softmax算子之前，首先需要导入PyPTO Pro、PyTorch和torch_npu模块。PyPTO Pro模块用于定义和编译Kernel，PyTorch和torch_npu模块用于构造输入数据、执行NPU计算并进行结果验证。

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu
```

同时定义本示例处理的数据Shape：

```python
ROWS = 8
COLS = 256
```

## 核心代码逻辑

1. 定义Tile类型和片上存储空间。

   PyPTO Pro SIMD编程中，需要显式定义计算过程中使用的Tile类型。本示例中，输入、中间结果和输出Tile的Shape为[8, 256]，最大值和指数和的归约结果Tile的Shape为[8, 1]。

   ```python
   tile_type = pl.TileType(
       shape=[ROWS, COLS],
       dtype=pl.DT_FP32,
       target_memory=pl.MemorySpace.Vec,
   )

   reduce_type = pl.TileType(
       shape=[ROWS, 1],
       dtype=pl.DT_FP32,
       target_memory=pl.MemorySpace.Vec,
       layout=pl.DN,
   )
   ```

   使用pypto_pro.language.make_tile_group为不同计算阶段分配Vector侧UB空间。不同Tile Group使用不同片上地址，避免存储空间重叠。

   ```python
   src_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x0000,
       mutex_ids=[0],
   )

   tmp_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x2000,
       mutex_ids=[1],
   )

   work_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x4000,
       mutex_ids=[2],
   )

   out_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x6000,
       mutex_ids=[3],
   )

   max_group = pl.make_tile_group(
       type=reduce_type,
       addrs=0x8000,
       mutex_ids=[4],
   )

   sum_group = pl.make_tile_group(
       type=reduce_type,
       addrs=0x8100,
       mutex_ids=[5],
   )
   ```

2. 实现Softmax Kernel函数。

   通过pypto_pro.language.jit装饰器定义Softmax Kernel函数，并通过auto_mutex=True开启Tile Mutex自动管理。

   Softmax计算依次完成数据搬入、最大值归约、减最大值、指数计算、求和归约、归一化以及数据搬出。

   ```python
   @pl.jit(auto_mutex=True)
   def softmax_simd_kernel(
       src: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
       dst: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
   ):
       tile_type = pl.TileType(
           shape=[ROWS, COLS],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
       )

       reduce_type = pl.TileType(
           shape=[ROWS, 1],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
           layout=pl.DN,
       )

       src_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x0000,
           mutex_ids=[0],
       )

       tmp_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x2000,
           mutex_ids=[1],
       )

       work_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x4000,
           mutex_ids=[2],
       )

       out_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x6000,
           mutex_ids=[3],
       )

       max_group = pl.make_tile_group(
           type=reduce_type,
           addrs=0x8000,
           mutex_ids=[4],
       )

       sum_group = pl.make_tile_group(
           type=reduce_type,
           addrs=0x8100,
           mutex_ids=[5],
       )

       with pl.section_vector():
           src_tile = src_group.current()
           tmp_tile = tmp_group.current()
           work_tile = work_group.current()
           out_tile = out_group.current()
           max_tile = max_group.current()
           sum_tile = sum_group.current()

           # 1. GM -> UB
           pl.load(src_tile, src, [0, 0])

           # 2. 计算每行最大值
           pl.maximum(max_tile, src_tile, tmp_tile, dim=0)

           # 3. x - max(x)
           pl.expand_sub(work_tile, src_tile, max_tile, dim=0)

           # 4. exp(x - max(x))
           pl.exp(work_tile, work_tile)

           # 5. sum(exp(x - max(x)))
           pl.sum(sum_tile, work_tile, tmp_tile, dim=0)

           # 6. exp(x - max(x)) / sum(exp(x - max(x)))
           pl.expand_div(out_tile, work_tile, sum_tile, dim=0)

           # 7. UB -> GM
           pl.store(dst, out_tile, [0, 0])
   ```

   其中，pypto_pro.language.maximum和pypto_pro.language.sum沿最后一维执行归约。对于Shape为[8, 256]的输入，两次归约结果的Shape均为[8, 1]。

   pypto_pro.language.expand_sub和pypto_pro.language.expand_div将Shape为[8, 1]的归约结果广播至[8, 256]，分别完成减最大值和概率归一化。

## 测试用例

为了验证Softmax算子的正确性，使用PyTorch Tensor作为输入，通过PyPTO Pro Kernel进行计算，并与PyTorch内置Softmax函数的结果进行对比。在开始执行PyPTO Pro和PyTorch相关代码之前，需要指定对应的Device ID。

```python
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

torch.manual_seed(0)

src = torch.randn(
    ROWS,
    COLS,
    dtype=torch.float32,
    device=device,
)

dst = torch.empty_like(src)

softmax_simd_kernel(src, dst)

torch.npu.synchronize()

golden = torch.softmax(src, dim=-1)

torch.testing.assert_close(
    dst,
    golden,
    rtol=1e-4,
    atol=1e-5,
)

print("SIMD Softmax kernel passed!")
```

## 编译与执行

在已安装PyPTO Pro的环境中运行：

```bash
# 配置CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备ID
export TILE_FWK_DEVICE_ID=0

# 执行脚本
python3 softmax_simd.py
```

程序执行成功后，显示以下信息：

```text
SIMD Softmax kernel passed!
```
