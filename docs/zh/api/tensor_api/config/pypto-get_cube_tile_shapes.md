# pypto.get\_cube\_tile\_shapes

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

获取cube计算中设置的TileShape大小、以及多核切K功能的开关使能。

## 函数原型

```python
get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool]
```

## 参数说明

无参数

## 返回值说明

返回包含m, k, n方向上的TileShape大小、以及是否开启多核切K功能。

## 约束说明

无。

## 调用示例

```python
pypto.get_cube_tile_shapes()
```
