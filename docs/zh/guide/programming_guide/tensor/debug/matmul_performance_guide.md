# Matmul性能调优指导

PyPTO为用户提供了一套高效便捷的算子开发框架，在性能优化方面，PyPTO提供了丰富的可选配置，包括算子的tile配置、算法配置以及图优化配置等，这些配置项一方面为用户提供了极大的开发灵活度，另一方面也大大提高了用户的使用门槛。

本教程旨在为用户提供一套分析、优化Matmul算子的方法，助力用户在使用PyPTO进行算子开发时达到更好的综合性能。

## 性能优化目标

分析理论性能上限是开展算子性能优化的第一步。我们知道，对于任意算子或算法，完成某种运算的过程一般涉及数据的搬运和数据的运算两大部分，业界常采用算数强度来衡量这两大部分的比例关系，这也是我们理解算子理论性能上限的重要依据。

算数强度（Arithmetic Intensity）反映了一个算子或算法在客观理论层面的计算量与数据访问量的比例关系，定义为总浮点运算次数（FLOPs）除以总内存访问字节数，单位`FLOPs/Byte`。对于指定的硬件平台，定义其峰值算力与峰值带宽的比为算力带宽比（显然其单位与算数强度相同），则当算数强度大于算力带宽比时，可以认为当前算法在该硬件平台上的性能受限于计算能力（也就是Compute Bound），反之则是受限于内存带宽（也就是Memory Bound）。

下面进一步分析Matmul算子在指定硬件平台上的性能上限。

考虑一个普通矩阵乘法的数据搬入量（一般称为MTE2搬运量）与计算量，则满足Compute Bound的条件为：
$$
\frac{CP}{BW} \leq \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
上式中，$CP$、$BW$表示指定硬件平台的峰值算力和带宽，$mL1$、$nL1$表示在M、N轴的切分大小，也就是用户在调用`pypto.set_cube_tile_shapes`中需要指定的切分大小。

*备注：上述理论公式仅考虑最基本的纯内积算法，并暂时忽略了数据搬出的开销以简化分析。*

可以看出，切分大小一方面直接决定了Matmul算子切分后的任务数量，从而决定了实际执行时的分核数、计算轮次。另一方面，切分大小从理论层面决定了Matmul算子的算数强度。

因此，优化Matmul性能的第一步便是优化Tile配置。

## Tile配置优化

### 增大算数强度

当M、N足够大时（如训练场景），Matmul切分前的算数强度足够大并且切分后的任务数足够分满核，此时我们倾向于在M、N两个维度上都选择较大的切分配置，从而达到尽可能大的切分后算数强度，以期望达成Compute Bound。

直观感受是，切分越大，算数强度越大，Matmul计算越容易达到Compute Bound。这是由于，切分必然会引入重复搬运，切分越多重复搬运量越大，从而算数强度越低。而另一方面，切分大小又受片上多级缓存空间（L1、L0）的限制而不能无限增加。

记切分大小为：

```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1], enable_split_k=False)
```

其中，mL0、kL0和nL0表示在L0 Buffer的切分大小，mL1、kL1和nL1表示在L1 Buffer的切分大小。

由于NPU的CUBE计算是以分型数据块为最小计算粒度，因此，要求切分大小也要满足分型格式的要求（也就是外轴16元素对齐，内轴32字节对齐）。同时，切分大小还要满足Buffer空间约束，具体计算约束详见pypto.set_cube_tile_shapes。

以Atlas A3系列产品和Atlas A2系列产品为例，对于A、B矩阵均为FP16类型的场景，满足Buffer空间约束的推荐Tile配置为：

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=False)
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128], enable_split_k=False)
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128], enable_split_k=False)
```

以上Tile配置的优点：

- 在满足L0 Buffer约束的条件下可以达到较大的算数强度；

$$
AI = \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
根据上式，显然$mL1 = nL1$时AI取到极大值，又由于$mL1 * nL1 * sizeof(float) <= L0C\_SIZE = 131072$，则当$mL1 = nL1 = sqrt(L0C\_SIZE / sizeof(float)) = 181$ 时，AI极大值成立。但是由于Tile大小需要满足分型格式的对齐要求，同时要考虑切分大小对于写入、写出带宽的影响，一般取128-256的组合。

- MTE2、MTE1搬运均可以开启double buffer，可以使能流水并行；
 上述配置下，L0A、L0B的空间占用为32KB，刚好可以使能MTE1 double buffer，同理，上述tile配置下MTE2也可以使能double buffer；同时由于kL1 > kL0并使能了大包搬运，单次MTE2的搬运量可以进一步增加，这有利于提高MTE2的带宽利用率。
 不过需要注意，当mL1与nL1取128-256组合时，L0C上无法开启nbuffer，因此这种tile配置适用于K轴较大（即搬出次数相对较少）的场景。当需要频繁搬出时，可以考虑选择128-128组合。

需要特别说明的是，上述Tile配置并非一成不变，需要用户根据计算场景（考虑输入shape、dtype、format等）以及硬件平台进行综合考虑。实际上，如果把Matmul看作一个算法，那么Tile的配置（Tiling）则是这个算法中最核心的部分。

同时，达到Compute Bound不仅需要提高算数强度，还需要尽可能提高访存带宽，这方面内容在后续继续介绍。

#### 减少重复载入

当M、N中有某个维度相对较小（如推理场景）时，Matmul在切分前的算数强度相对较小，一般只能达到Memory Bound。此时我们的优化思路主要是在保证分满核的前提下，尽量减少数据重复载入（主要是MTE2重复载入）。

一般而言，我们至少要保证分核数达到总核数的0.8以上（以24核平台为例，需要至少分20核），这样可以保证MTE2带宽具有较高的利用率。用公式表示如下：
$$
\frac{M}{mL1} \cdot \frac{N}{nL1} >= 0.8 \cdot coreNum
$$
在此基础上，需要尽可能减小重复载入量。当K轴存在L1切分时，MTE2总载入量为：
$$
MTE2\_LOAD\_SIZE = M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte
$$
综上，此时的Tile配置求解就转化为一个约束条件下的极值求解问题。

举例，对于输入规格为M = 96，K = 1536，N = 3072，数据类型为FP16的场景。
为了避免B矩阵的MTE2重复载入，一种较好的切分方式是设置mL1 = M = 96；另外，为了分满核，推荐设置nL1 = N / coreNum = 128。在此基础上，进一步考虑MTE2、MTE1的流水并行以及MTE2带宽利用率，设置kL1 = 4kL0。
综上，一个较好的Tile配置是：

```python
pypto.set_cube_tile_shapes([96, 96], [64, 256], [128, 128], enable_split_k=False)
```

进一步优化，可以将A矩阵一次性搬入L1中，然后一直驻留并反复使用。这样MTE2总载入量可以进一步减小至：
$$
MTE2\_LOAD\_SIZE = M \cdot K \cdot aByte + K \cdot N \cdot bByte
$$
这样可以完全消除MTE2重复载入，进一步优化整体性能。此时的Tile配置如下：

```python
pypto.set_cube_tile_shapes([96, 96], [64, 1536, 256], [128, 128], enable_split_k=False)
```

这里使用了一个相对高级的Tile配置用法，即独立设置kAL1与kBL1，形式如下：

```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1], enable_split_k=False)
```

此处kAL1和kBL1分别表示A矩阵和B矩阵的K轴在L1上的切分大小。当K轴只设置两个切分值时，表示kAL1 = kBL1 = kL1。

#### K轴分核

对于M、N较小而K轴较大的场景，仅在M、N轴做分核可能无法用满核导致整体性能较差，此时可以采用K轴分核策略进行优化。

多核切K有两种使能方式：

- **手动构图方式**：手动对K轴进行切分，并调用`pypto.matmul`以及`pypto.add`完成多核切K构图。该方式可独立配置单核切分长度以及Vector的切分大小，适合深度性能调优场景。
- **自动构图方式**：采用`enable_split_k`开关使能多核切K，适合快速验证场景，但是**不保证性能最优**。

##### 手动构图方式

示例代码如下：

```python
@pypto.frontend.jit
def matmul_demo_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
    pypto.set_vec_tile_shapes(32, 256) # 配置add切分大小

    partial_sum = []
    # 使用前需提前定义k_size（一般是a矩阵的k维度值）和k_view_size（k轴view shape大小）的大小
    k_size = a.shape[1] # 假设a 矩阵不转置
    k_view_size = 32
    k_loop = (k_size + k_view_size - 1) // k_view_size # 配置单核累加长度
    for k_idx in range(k_loop):
        a_view = a[:, k_idx * k_view_size: k_idx * k_view_size + k_view_size]
        b_view = b[k_idx * k_view_size: k_idx * k_view_size + k_view_size, :]
        res = pypto.matmul(a_view, b_view, pypto.DT_FP32)
        partial_sum.append(res)

    for i in range(1, len(partial_sum)):
        partial_sum[0] += partial_sum[i]

    out[:] = partial_sum[0]
```

##### 自动构图方式

对于快速验证场景，可以通过`enable_split_k=True`快速使能K轴分核：

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
```

此时默认采用`kL1`对K轴进行切分，单核内计算`kL1`长度的部分和后搬出，最后累加所有部分和。代码实现如下：

```python
@pypto.frontend.jit
def matmul_split_k_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
    out[:] = pypto.matmul(a, b, pypto.DT_FP32)
```

## 访存带宽优化

### 增加L2命中率

回到Compute Bound判据公式：
$$
\frac{CP}{BW} \leq \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
前文的优化措施主要围绕增大算数强度展开，本节将主要聚焦如何提高带宽利用率以缩小算力带宽比。考虑L2命中率对于综合带宽的影响，对纯内积算法而言，总的MTE2搬运量计算公式为：
$$
MTE2\_TOTAL\_LOAD\_SIZE = M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte
$$
其中从HBM和L2中搬运的数据量分别为：
$$
HBM\_LOAD\_SIZE = M \cdot K \cdot aByte + K \cdot N \cdot bByte
$$
$$
L2\_LOAD\_SIZE = MTE2\_TOTAL\_LOAD\_SIZE - HBM\_LOAD\_SIZE
$$
记L2命中率为L2载入量在总载入量中的占比：
$$
l2\_hit\_ratio = \frac{L2\_LOAD\_SIZE}{MTE2\_TOTAL\_LOAD\_SIZE}=1-\frac{\frac{1}{N} \cdot aByte+\frac{1}{M} \cdot bByte}{\frac{1}{nL1} \cdot aByte+\frac{1}{mL1} \cdot bByte}
$$
对于Atlas A3系列产品和Atlas A2系列产品，L2带宽是HBM带宽的3倍以上，因此需要优先提高L2命中率。

进一步考虑A、B矩阵数据类型均为FP16的场景，并考虑单轮次的L2命中率，则有：
$$
l2\_hit\_ratio = 1-\frac{\frac{1}{nDim \cdot nL1} +\frac{1}{mDim \cdot mL1} }{\frac{1}{nL1} +\frac{1}{mL1} }
$$
上式中，$mDim, nDim$分别为单轮次计算时M、N轴的分核数。则对于指定切分配置，显然当$nDim \cdot nL1 = mDim \cdot mL1$ 时，单轮次的L2命中率最大。

综合考虑算数强度与L2命中率，以Atlas A3系列产品和Atlas A2系列产品为例，为了尽可能提高算数强度，我们一般取128-256切分配置，以mL1=128，nL1=256为例，则单轮次M、N轴的分核数满足$nDim \cdot 256 = mDim \cdot 128$时L2命中率最高，对于24核平台，可以取mDim=6，nDim=4或mDim=8，nDim=3。

举例，对于M = N = K = 6144，FP16类型场景，仅采用128-256切分配置，整体耗时约为2.1ms，等效算力约220 TFLOPS，测试代码如下：

```python
import torch
import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0})
def matmul_native_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
    out[:] = pypto.matmul(a, b, pypto.DT_FP16)


def test_native_mm():
    M, K, N = 6144, 6144, 6144
    a = torch.randn([M, K], dtype=torch.float16, device="npu:0")
    b = torch.randn([K, N], dtype=torch.float16, device="npu:0")
    out = torch.empty(M, N, dtype=torch.float16, device="npu:0")
    matmul_native_kernel(a, b, out)


if __name__ == "__main__":
    test_native_mm()
```

在上述切分配置的基础上进一步优化L2命中率，整体耗时约1.6ms，等效算力约290 TFLOPS，测试代码如下：

```python
import torch
import pypto


M, K, N = 6144, 6144, 6144
mL1, nL1 = 128, 256
mDim, nDim = 6, 4
m_view = mL1 * mDim   # 768
n_view = nL1 * nDim   # 1024


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0})
def matmul_l2_split_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])

    m_loop = (M + m_view - 1) // m_view
    n_loop = (N + n_view - 1) // n_view
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            a_view = a[m_idx * m_view : m_idx * m_view + m_view, :]
            b_view = b[:, n_idx * n_view : n_idx * n_view + n_view]
            out_view = pypto.matmul(a_view, b_view, pypto.DT_FP16)
            out[m_idx * m_view : m_idx * m_view + m_view, n_idx * n_view : n_idx * n_view + n_view] = out_view


def test_mm_with_l2_split():
    a = torch.randn([M, K], dtype=torch.float16, device="npu:0")
    b = torch.randn([K, N], dtype=torch.float16, device="npu:0")
    out = torch.empty(M, N, dtype=torch.float16, device="npu:0")
    matmul_l2_split_kernel(a, b, out)


if __name__ == "__main__":
    test_mm_with_l2_split()
```
