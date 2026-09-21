# pypto_pro.language.get_subblock_num

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

获取当前block的subblock总数（即一个block关联的从核数量，又称task ration）。

## 函数原型

```python
pypto_pro.language.get_subblock_num() -> int
```

## 参数说明

无。

## 约束说明

无。

## 返回值说明

返回DT_INT64类型的subblock总数。返回值与核类型及编译模式有关：

- **AIC核**：始终返回1（AIC为block，无AIC从核）。
- **AIV核**：
  - 融合算子（mix，AIC:AIV = 1:2）：返回2（每个AI Core含2个AIV从核）。
  - 纯Vector算子（aiv-only）：返回1（AIV为block，无subblock划分）。

## 调用示例

在融合算子中，get_block_idx()在AIV核上返回的是逻辑编号（block_idx * subblock_num + subblock_idx），通过除以get_subblock_num()可还原物理AI Core编号，使cube与vector两侧用统一的core_id切分数据：

```python
import pypto_pro.language as pl

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def matmul_example(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    num_cores = pl.get_block_num()
    # AIC/AIV两侧得到相同的物理核号，详见下方NOTE
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    with pl.section_cube():
        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            ...  # Cube侧按行块i执行load/matmul/store
    with pl.section_vector():
        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            ...  # Vector侧用同一core_id切分，与Cube侧对齐


matmul_example[None, NUM_CORES](a, b, out)
```

> [!NOTE]说明
> 该除法在AIC核上为block_idx // 1，在AIV核上为(block_idx * 2 + subblock_idx) // 2，两者均得到相同的AI Core编号，因此cube与vector可共享同一core_id做数据切分。
