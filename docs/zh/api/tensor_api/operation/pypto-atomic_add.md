# pypto.atomic\_add

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

以offsets指定的dst索引位置为基准，将输入Tensor src以原子（Atomic）方式累加到输出Tensor dst的对应区域。以2维为例，计算公式如下：

$$
dst\left[ offsets\left[0\right] : offsets\left[0\right] + src.shape\left[0\right],\ offsets\left[1\right] : offsets\left[1\right] + src.shape\left[1\right] \right]\ += src
$$

其他维度以此类推。

## 函数原型

```python
atomic_add(src: Tensor, offsets: List[Union[int, SymbolicScalar]], dst: Tensor) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src     | 输入      | 源操作数，类型为Tensor。|
| offsets | 输入      | src写入dst时各维度的起始偏移量，元素类型为int或SymbolicScalar。 |
| dst     | 输出      | 目标操作数，类型为Tensor。 |

## 返回值说明

无返回值，结果直接写入dst（inplace操作）。

## 约束说明

1. 输出Tensor dst的valid shape需由用户在调用atomic_add前确保正确，该接口不会自动推导。

2. offsets的长度需等于src/dst的维度数，即 $len(offsets) == src.dim == dst.dim$，否则会产生编译报错。

3. 在调用atomic_add前，需保证dst对应区域的数据已有效，否则会导致未定义行为。

4. 为保证写入区域不越界，对任意维度 $i$ 需满足 $offsets[i] + src.shape[i] \le dst.shape[i]$，否则会导致未定义行为。

5. 当多个核并发对dst的重叠区域执行atomic_add时，各次累加的执行顺序不确定。

## 调用示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.atomic_add(x, [0, 0], out)
```

结果示例如下：

```txt
输入数据x: [[1, 1],
           [1, 1]]
输出数据out（运算前）: [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
输出数据out（运算后）: [[2, 2, 0, 0],
                      [2, 2, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
```
