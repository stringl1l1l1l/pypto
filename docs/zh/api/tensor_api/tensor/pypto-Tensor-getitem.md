# pypto.Tensor.\_\_getitem\_\_

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

通过索引或切片的方式从Tensor中获取子Tensor或单个元素等，该方法支持多种索引模式，提供了灵活且直观的数据访问方式。

## 函数原型

```python
__getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None)
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| key         | 输入      | Tensor索引，用于获取Tensor对应位置的数据。<br> 支持类型：<br> - int或SymbolicScalar（符号标量）: 单个整数索引。<br> - slice: 切片对象。<br> - tuple: 多维索引的组合，类型包括：int或SymbolicScalar，slice，Ellipsis(...)。 |
| valid_shape | 输入      | 表示输出Tensor有效数据的大小。 |

## 返回值说明

- 当索引全部为整数（int或SymbolicScalar）时，返回对应位置的单个元素，类型为SymbolicScalar。
- 当索引中包含slice时，返回对应索引位置的子Tensor。

## 约束说明

1. 对于slice（切片对象，格式为start:end:step），当前功能暂时不支持step设置,值默认固定为1。

   未支持示例：a\[1:2:2, :\]。

2. 当前功能暂时不支持bool类型索引。

   未支持示例：a\[True, False, True, False\]。

3. 当前功能暂时不支持Tensor类型索引。

   未支持示例：a\[b\]，\(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)\)。

4. 省略号（...）在索引中最多出现一个，且索引个数不能超过Tensor的维度数。

## 调用示例

1. 空切片（slice）

   使用空切片获取Tensor整体，返回自身。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:] #返回a自身
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   ```

2. 全切片（slice）

   使用切片获取Tensor的子区域

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:2, :2] #等价于view(a, [2, 2], [0, 0])
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [[1, 2],
               [5, 6]]
   ```

3. 缺省索引

   索引个数小于Tensor维度数时，自动为缺失的维度补全全切片（:）。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1] #等价于a[1, :]
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [5, 6, 7, 8]
   ```

4. 混合索引和切片

   结合整数索引和切片，可以降低维度并提取特定行或列。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1, 1:3] #等价于先view(a, [1, 2], [1, 1])，再reshape为 [2]
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [6, 7]
   ```

5. 负数索引

    支持Python风格的负索引，从末尾开始计数。

    ```python
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    b = a[-1, -3:-1] #等价于a[3, 1:3]
    ```

    结果示例如下：

    ```python
    输入数据a: [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
    输出数据b: [14, 15]
    ```

6. 省略号（...）

   使用 \`...\`自动填充中间的所有维度，简化多维索引。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[..., 1:3] #等价于a[:, 1:3]
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [[2, 3],
               [6, 7],
               [10, 11],
               [14, 15]]
   ```

7. 单元素访问

   整数索引，取出Tensor的单个元素（仅支持DT\_INT32类型）。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_INT32)
   b = a[0, 0] #返回SymbolicScalar
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: 1
   ```

8. Gather操作

   当索引为\[int:Tensor\]的形式时，执行gather操作，索引中int类型对应dim，Tensor类型对应index，该切片语法等价于Tensor.gather\(dim, index\)。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   index = pypto.tensor([1, 4], pypto.DT_INT32)
   b = a[0:index] #调用gather(a, 0, index)
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输入数据index: [[0, 1, 2, 3]]
   输出数据b: [[1, 6, 11, 16]]
   ```
