# Softmax算子快速入门（SIMT）

## 任务与目标

本节将详细介绍如何使用PyPTO Pro SIMT编程范式实现一个简单的Softmax算子，并通过测试用例验证其正确性。通过本节的学习，您将了解如何使用PyPTO Pro SIMT相关API完成线程级并行计算、线程索引获取以及原子归约。

本示例固定处理Shape为[1, 256]、数据类型为FP32的Tensor，显式定义线程级计算逻辑并启动256个SIMT线程，由每个线程负责一个输入元素，协同完成Softmax中的最大值归约和求和归约。

## 算子设计规格

**表1** Softmax算子设计规格

| name            | shape    | data type | format |
| --------------- | -------- | --------- | ------ |
| **inputs（输入）**  | (1, 256) | float32   | ND     |
| **outputs（输出）** | (1, 256) | float32   | ND     |

* 数学表达式

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i-M)}{\sum_j \exp(z_j-M)}$

* 使用的主要接口

  SIMT函数与线程启动接口：pypto_pro.language.simt.function、pypto_pro.language.simt.launch

  线程索引接口：pypto_pro.language.simt.thread_idx

  基础计算接口：pypto_pro.language.simt.exp

  原子操作接口：pypto_pro.language.simt.atomic_max、pypto_pro.language.simt.atomic_add

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

1. 实现核心计算函数。

   Softmax计算分为最大值归约、指数计算与求和归约、概率归一化三个阶段。每个SIMT线程通过线程索引访问对应的输入元素。

   最大值归约阶段，各线程通过pypto_pro.language.simt.atomic_max将自身负责的输入元素原子更新至公共最大值。

   ```python
   @pl.simt.function(max_threads=THREADS)
   def reduce_max(src, max_value):
       tid = pl.simt.thread_idx().x
       pl.simt.atomic_max(max_value[0, 0], src[0, tid])
   ```

   指数计算与求和归约阶段，每个线程首先计算exp(x - max(x))，然后通过pypto_pro.language.simt.atomic_add将计算结果累加至公共指数和。

   ```python
   @pl.simt.function(max_threads=THREADS)
   def exp_and_sum(src, exp_value, max_value, sum_value):
       tid = pl.simt.thread_idx().x

       value = pl.simt.exp(
           src[0, tid] - max_value[0, 0]
       )

       exp_value[0, tid] = value

       pl.simt.atomic_add(
           sum_value[0, 0],
           value,
       )
   ```

   归一化阶段，每个线程将对应位置的指数结果除以公共指数和，得到最终Softmax输出。

   ```python
   @pl.simt.function(max_threads=THREADS)
   def normalize(exp_value, dst, sum_value):
       tid = pl.simt.thread_idx().x

       dst[0, tid] = (
           exp_value[0, tid]
           / sum_value[0, 0]
       )
   ```

2. 实现Softmax Kernel函数。

   为了保证最大值归约、指数求和和归一化三个阶段之间的数据依赖，本示例分别定义三个JIT Kernel。每个Kernel通过pypto_pro.language.simt.launch启动256个SIMT线程。

   最大值归约Kernel如下：

   ```python
   @pl.jit(arch="a5")
   def reduce_max_kernel(
       src: pl.Tensor[[1, THREADS], pl.DT_FP32],
       max_value: pl.Tensor[[1, 1], pl.DT_FP32],
   ):
       with pl.section_vector():
           pl.simt.launch(
               reduce_max,
               threads=THREADS,
               args=(src, max_value),
           )
   ```

   指数计算和求和归约Kernel如下：

   ```python
   @pl.jit(arch="a5")
   def exp_sum_kernel(
       src: pl.Tensor[[1, THREADS], pl.DT_FP32],
       exp_value: pl.Tensor[[1, THREADS], pl.DT_FP32],
       max_value: pl.Tensor[[1, 1], pl.DT_FP32],
       sum_value: pl.Tensor[[1, 1], pl.DT_FP32],
   ):
       with pl.section_vector():
           pl.simt.launch(
               exp_and_sum,
               threads=THREADS,
               args=(
                   src,
                   exp_value,
                   max_value,
                   sum_value,
               ),
           )
   ```

   归一化Kernel如下：

   ```python
   @pl.jit(arch="a5")
   def normalize_kernel(
       exp_value: pl.Tensor[[1, THREADS], pl.DT_FP32],
       dst: pl.Tensor[[1, THREADS], pl.DT_FP32],
       sum_value: pl.Tensor[[1, 1], pl.DT_FP32],
   ):
       with pl.section_vector():
           pl.simt.launch(
               normalize,
               threads=THREADS,
               args=(
                   exp_value,
                   dst,
                   sum_value,
               ),
           )
   ```

   三个Kernel依次完成：

   ```text
   reduce_max_kernel
           ↓
      max(x)
           ↓
   exp_sum_kernel
           ↓
   exp(x - max(x))
           ↓
   sum(exp(x - max(x)))
           ↓
   normalize_kernel
           ↓
      Softmax
   ```

## 测试用例

为了验证Softmax算子的正确性，使用PyTorch Tensor作为输入，通过PyPTO Pro Kernel进行计算，并与PyTorch内置Softmax函数的结果进行对比。

最大值归约、指数求和和归一化三个Kernel依次执行，并通过torch.npu.synchronize保证不同计算阶段之间的执行顺序。

```python
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"

torch.npu.set_device(device)

torch.manual_seed(0)

src = torch.randn(
    1,
    THREADS,
    dtype=torch.float32,
    device=device,
)

exp_value = torch.empty_like(src)
dst = torch.empty_like(src)

max_value = torch.full(
    (1, 1),
    -3.4028235e38,
    dtype=torch.float32,
    device=device,
)

sum_value = torch.zeros(
    (1, 1),
    dtype=torch.float32,
    device=device,
)

reduce_max_kernel(
    src,
    max_value,
)

torch.npu.synchronize()

exp_sum_kernel(
    src,
    exp_value,
    max_value,
    sum_value,
)

torch.npu.synchronize()

normalize_kernel(
    exp_value,
    dst,
    sum_value,
)

torch.npu.synchronize()

golden = torch.softmax(
    src,
    dim=-1,
)

torch.testing.assert_close(
    dst,
    golden,
    rtol=1e-4,
    atol=1e-5,
)

print("SIMT Softmax kernel passed!")
```

其中，max_value初始化为较小的FP32数值，用于最大值归约；sum_value初始化为0，用于通过原子加完成指数和归约。

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
