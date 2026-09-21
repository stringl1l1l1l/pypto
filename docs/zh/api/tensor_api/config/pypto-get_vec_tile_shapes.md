# pypto.get\_vec\_tile\_shapes

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

获取vector计算中的TileShape大小。

## 函数原型

```python
get_vec_tile_shapes() -> List[int]
```

## 参数说明

void

## 返回值说明

返回每个维度的TileShape大小。

## 约束说明

无。

## 调用示例

```python
pypto.get_vec_tile_shapes()
```
