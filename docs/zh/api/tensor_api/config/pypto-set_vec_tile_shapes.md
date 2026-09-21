# pypto.set\_vec\_tile\_shapes

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

设置vector计算中的TileShape大小。

## 函数原型

```python
set_vec_tile_shapes(*args: int) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                  |
|---------|-----------|-----------------------|
| *args   | 输入      | 每个维度的TileShape大小，最多不超过4个inputs |

## 返回值说明

void

## 约束说明

TileShape需要满足以下约束条件：

每个维度必须大于0。

假设TileShape是二维\{m, n\}，则：

- （m \> 0）&& \(n \> 0\)

## 调用示例

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)
```
