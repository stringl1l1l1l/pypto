# Tensor的操作

## 支持数学运算

PyPTO为张量计算提供了一套全面的操作，旨在为用户提供高效、灵活的计算能力。这些操作分为向量操作、矩阵操作，并将相关操作进行组合以实现更复杂的计算逻辑。

### 向量运算

向量运算是在张量上执行元素级别的计算，适用于各种基本数学操作。

- 算术运算

    ```python
    # 加法
    result = pypto.add(a, b)
    result = pypto.add(a, scalar)  # 将标量添加到张量
    result = a + b

    # 减法
    result = pypto.sub(a, b)
    result = pypto.sub(a, scalar)
    result = a - b

    # 乘法
    result = pypto.mul(a, b)
    result = pypto.mul(a, scalar)
    result = a * b

    # 除法
    result = pypto.div(a, b)
    result = pypto.div(a, scalar)
    result = a / b

    # 指数
    result = pypto.pow(a, scalar)  # a ** scalar
    ```

- 数学函数

    ```python
    # 指数和对数
    result = pypto.exp(x)      # e ** x
    result = pypto.log(x)      # ln(x)
    result = x.exp()
    result = x.log()

    # 开根号
    result = pypto.sqrt(x)     # √x
    result = pypto.rsqrt(x)    # 1/√x
    result = x.sqrt()
    result = x.rsqrt()

    # 三角函数
    result = pypto.sin(x)
    result = pypto.cos(x)
    result = x.sin()
    result = x.cos()

    # 绝对值
    result = pypto.abs(x)
    result = x.abs()

    # 相反数
    result = pypto.neg(x)      # -x
    result = x.neg()
    ```

- 激活函数

    ```python
    # Sigmoid
    result = pypto.sigmoid(x)  # 1 / (1 + exp(-x))
    result = x.sigmoid()

    # ReLU
    result = pypto.relu(x)     # 最大值(0, x)
    result = x.relu()

    # Softmax
    result = pypto.softmax(x, dim=-1)  # 沿着dim维度做Softmax
    result = x.softmax(dim=-1)
    ```

- 比较操作

    ```python
    # 最大值和最小值
    result = pypto.maximum(a, b)  # 逐元素最大值
    result = pypto.minimum(a, b)  # 逐元素最小值

    # 截断
    result = pypto.clip(x, min_val, max_val)  # 将x的取值范围进行截断
    ```

- 归约操作

    ```python
    # 求和
    result = pypto.sum(x, dim=-1, keepdim=False)
    result = x.sum(dim=-1, keepdim=False)

    # 最大值
    result = pypto.amax(x, dim=-1, keepdim=False)
    result = x.amax(dim=-1, keepdim=False)

    # 最小值
    result = pypto.amin(x, dim=-1, keepdim=False)
    result = x.amin(dim=-1, keepdim=False)
    ```

- 原地修改 \(inplace\)

    ```python
    # 使用move()进行就地操作
    output.move(pypto.add(a, b))  # 高效，无拷贝

    # 或者使用赋值
    output[:] = pypto.add(a, b)   # 也是高效的
    ```

- 广播模式（broadcast）

    许多操作支持广播，这使得操作更加灵活和高效。

    ```python
    # 张量+标量（张量为标量）
    result = pypto.add(tensor, 2.0)

    # 张量+一维张量（一维张量）
    bias = pypto.tensor([features], pypto.DT_BF16, "bias")
    result = pypto.add(tensor, bias)
    ```

### 矩阵运算

矩阵运算针对NPU的Cube核进行了优化，适用于大规模矩阵计算。

```python
# 基本矩阵乘法
# C = A @ B，其中A: [M, K], B: [K, N], C: [M, N]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

# 用转置，其中A: [M, K], B: [N, K]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True)

# 批量矩阵乘法
# A: [B, M, K], B: [B, K, N]，结果：[B, M, N]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

# 带有偏置
bias = pypto.tensor([1, N], pypto.DT_BF16, "bias")
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, extend_params={'bias_tensor': bias})

# 用转置，其中A: [M, K], B: [N, K],输出为NZ格式
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True, c_matrix_nz=True)
```

矩阵乘法参数：

- `input`：左矩阵 \[M, K\] 或 \[B, M, K\]
- `mat2`：右矩阵 \[K, N\] 或 \[B, K, N\]
- `out_dtype`：输出数据类型
- `a_trans`：转置左矩阵（默认：False）
- `b_trans`：转置右矩阵（默认：False）
- `c_matrix_nz`：以NZ格式输出（默认值：False）
- `extend_params`：扩展特征（偏置、去量化等）。

### 组合操作

可以组合上述基本操作，以实现更复杂的计算逻辑。

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    row_max = pypto.amax(x, dim=-1, keepdim=True)  # 计算行最大值
    sub = x - row_max                              # 值归一化
    exp = pypto.exp(sub)                           # 指数运算
    esum = pypto.sum(exp, dim=-1, keepdim=True)    # 求和
    return exp / esum                              # 概率归一化

def softmax_kernel(x: pypto.Tensor, y: pypto.Tensor) -> None:
    ...
    for idx in pypto.loop(x.shape[0]):
        ...
        softmax_out = softmax_core(x)
        ...
```

## 支持逻辑结构变换

逻辑结构变换操作允许用户在张量上进行形状、维度和类型等变换，以适应不同的计算需求。

### 视图和组装

- 视图：视图操作创建对相同基础数据的新张量引用，适用于分块处理和局部计算。

    ```python
    # 视图
    view = pypto.view(tensor, view_shape, offset, valid_shape)
    ```

    示例：

    ```python
    # 创建具有特定形状和偏移量的视图
    view = pypto.view(
        tensor,
        view_shape=[32, 32],
        offset=[10, 20],
        valid_shape=[actual_h, actual_w]  # 可选
    )
    ```

    参数说明：

    - tensor：源张量
    - view\_shape：视图的形状
    - offset：源张量的起始位置
    - valid\_shape：实际有效尺寸（用于边界处理）

    Tensor支持Python风格的索引和切片，适用于灵活的数据访问。

    ```python
    tensor = pypto.tensor([10, 20], pypto.DT_FP16, "tensor")

    # 单个元素（创建视图）
    element = tensor[0, 0]  # 只支持INT32

    # 切片（创建视图）
    slice_tensor = tensor[0:5, 10:20]

    # 省略号
    ellipsis_slice = tensor[..., 0:10]
    ```

- 组装：assemble函数在指定的偏移处将较小的张量放入较大的张量，适用于分块处理后的结果合并。

    ```python
    # 将一个小的张量组装成一个大的张量
    pypto.assemble(
        small_tensor,      # 源张量
        [10, 20],          # 目标位置
        large_tensor       # 目标张量
    )
    ```

    示例：

    ```python
    # 小张量结果
    tile_result = pypto.tensor([32, 32], pypto.DT_FP16, "tile")

    # 大输出张量
    output = pypto.tensor([100, 200], pypto.DT_FP16, "output")

    # 在 [10, 20] 位置组装输出张量
    pypto.assemble(tile_result, [10, 20], output)
    ```

### 重塑形状/维度

```python
# 重塑张量
reshaped = pypto.reshape(tensor, [new_shape])

# 转置
transposed = pypto.transpose(tensor, dim0=0, dim1=1)
```

重塑不会改变数据，只会改变张量的视图。

### 类型转换

```python
# 转换为不同的数据类型
result = pypto.cast(tensor, pypto.DT_FP32, mode=pypto.CastMode.CAST_NONE)
```
