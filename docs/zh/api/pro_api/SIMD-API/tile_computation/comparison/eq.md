# pypto_pro.language.eq

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

将lhs与rhs逐元素比较是否相等，生成bit-packed掩码。rhs可以是Tile或标量。掩码输出为DT_UINT8类型，通常配合[pypto_pro.language.select](../selection/select.md)使用。

## 函数原型

```python
pypto_pro.language.eq(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存储空间为UB，采用行主序排布。数据类型为DT_UINT8、DT_INT8、DT_BOOL，采用按位压缩格式，形状和有效形状必须与lhs一致。 |
| lhs | 输入 | 源操作数（左操作数），Tile类型，存储空间为UB，采用行主序排布，形状和有效形状必须与out一致。支持8、16、32、64位整型、DT_FP16、DT_BF16和DT_FP32。 |
| rhs | 输入 | 源操作数（右操作数），Tile或Scalar类型，也支持可转换为Scalar的Python int或float常量。传入Tile时，存储空间为UB，采用行主序排布，数据类型、形状和有效形状必须与lhs一致；传入Scalar或Python常量时，数据类型必须与lhs的元素类型兼容。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
pl.eq(mask, lhs, rhs)  # Tile-Tile比较：lhs == rhs
pl.eq(mask, lhs, 0)    # Tile-Scalar比较：lhs == 0
```
