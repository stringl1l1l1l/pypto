# pypto.Tensor.\_\_setitem\_\_

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

通过索引或切片的方式向Tensor的指定位置赋值。

## 函数原型

```python
__setitem__(self, key, value)
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| key     | 输入      | Tensor索引，用于定位Tensor中待赋值的位置。<br> 支持类型：<br> - int或SymbolicScalar（符号标量）: 单个整数索引。<br> - slice: 切片对象。<br> - tuple: 多维索引的组合，类型包括：int或SymbolicScalar，slice，Ellipsis(...)。 |
| value   | 输入      | 需要设置的值。<br> 支持类型：<br> - Tensor：切片或混合索引赋值时使用，其shape须与索引覆盖的区域匹配。<br> - 标量（int或SymbolicScalar）：单元素赋值时使用，且目标Tensor须为DT\_INT32类型。 |

## 返回值说明

无返回值，原地修改目标Tensor。

## 约束说明

1. 对于slice（切片对象，格式为start:end:step），当前功能暂时不支持step设置,值默认固定为1。

   未支持示例：a\[1:2:2, :\]。

2. 当前功能暂时不支持bool类型索引。

   未支持示例：a\[True, False, True, False\]。

3. 当前功能暂时不支持Tensor类型索引。

   未支持示例：a\[b\]，\(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)\)。

4. 混合索引（int/SymbolicScalar与slice组合）赋值时，key中slice的个数必须与value的维度数一致。

5. 单元素赋值时，value仅支持int或SymbolicScalar类型，暂不支持float类型。

## 调用示例

1. 空切片（slice）

   使用空切片对Tensor整体赋值，等价于将value整体移动到目标Tensor（move），常用于将计算结果赋给输出Tensor。当value与目标Tensor形状一致时，最终数据效果等价于全量assemble。

   ```python
   @pypto.frontend.jit()
   def add_kernel(
      a: pypto.Tensor[[], pypto.DT_INT32], b: pypto.Tensor[[], pypto.DT_INT32], c: pypto.Tensor[[], pypto.DT_INT32]
   ):
      pypto.set_vec_tile_shapes(32, 32)
      c[:] = a + b
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输入数据b: [[1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1]]
   输出数据c: [[2, 3, 4, 5],
               [6, 7, 8, 9],
               [10, 11, 12, 13],
               [14, 15, 16, 17]]
   ```

2. 全切片（slice）

   使用切片将一个小Tensor组装到大Tensor的指定位置。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[0:, 0:] = b #等价于assemble(b, (0, 0), a)
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [[10, 10]
               [10, 10]]
   输出数据a: [[10, 10, 0, 0],
               [10, 10, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

3. 混合索引和切片

   结合整数索引和切片，可以对特定行或列进行操作。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[0, 1:3] = b #b被reshape为(1, 2),等价于pypto.assemble(b, (0, 1), a)
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [10, 10]
   输出数据a: [[0, 10, 10, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

4. 负索引

   支持Python风格的负索引，从末尾开始计数。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[-1, -3:-1] = b #等价于a[3, 1:3]
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [10, 10]
   输出数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 10, 10, 0]]
   ```

5. 省略号（...）

   使用... 可以自动填充中间维度。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[..., 2:4] = b #等价于a[0:2, 2:4]
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [[10, 10]
               [10, 10]]
   输出数据a: [[0, 0, 10, 10],
               [0, 0, 10, 10],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

6. 单元素赋值

   整数索引，对单个元素赋值（仅支持DT\_INT32类型）。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_INT32)
   a[2, 3] = 5 #调用SetTensorData
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输出数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 5],
               [0, 0, 0, 0]]
   ```

7. Scatter操作

   当key为slice，key.start为int， key.stop为Tensor时\(a\[start:stop\]   \)，执行scatter操作。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   indices = pypto.Tensor([1, 4], pypto.DT_INT32) # 索引Tensor
   values = pypto.Tensor([1, 4], pypto.DT_FP32)
   # 在维度0上进行scatter
   a[0:indices] = values #调用pypto.scatter(a, 0, indices, values)
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据indices：[[0, 1, 2, 3]]
   输入数据values：[[10, 10, 10, 10]]
   输出数据a: [[10, 0, 0, 0],
               [0, 10, 0, 0],
               [0, 0, 10, 0],
               [0, 0, 0, 10]]
   ```
