# pypto_pro.language.min

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

取两个标量中的较小值。

## 函数原型

```python
pypto_pro.language.min(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 左操作数，为整型或浮点型常量，或运行时整型或浮点型标量表达式。 |
| rhs | 输入 | 右操作数，为整型或浮点型常量，或运行时整型或浮点型标量表达式。 |

## 约束说明

- lhs和rhs必须是数值标量。两个操作数同为整型或同为浮点型时，结果提升为位宽较大的数据类型；一个为整型、另一个为浮点型时，两个操作数均转换为FP32后再比较。
- 接口只接受两个位置参数，不支持关键字参数。
- 适用范围：本接口仅用于标量取较小值。Tile逐元素取最小值请使用pypto_pro.language.minimum。

## 返回值说明

返回两个标量中的较小值，Scalar类型。返回值数据类型由参数的数据类型提升结果决定。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl

# 循环边界计算（尾块处理）
m_size = pl.min(m_dim - i, 64)
n_size = pl.min(n_dim - j, 128)

# 与整数常量比较
causal_kv_tiles = pl.min(qi + 1, skv_tiles)

# 标量变量之间
c = pl.min(a, b)

# 浮点标量
scale = pl.min(pl.const(2.5, pl.DT_FP32), pl.const(-1.5, pl.DT_FP32))
```
