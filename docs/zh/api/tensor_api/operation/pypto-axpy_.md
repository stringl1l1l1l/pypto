# pypto.axpy_

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

执行AXPY操作：`y = alpha * x + y`。该操作对y张量进行原地更新。

计算公式如下：

$$
y_i = \alpha \cdot x_i + y_i
$$

**重要说明**：AXPY是原地操作，y张量会被直接修改。如果后续计算需要使用执行AXPY之前的原始y值，请在调用AXPY前使用`pypto.clone(y)`进行备份。

## 函数原型

```python
axpy_(y: Tensor, x: Tensor, alpha: Union[int, float] = 1.0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| y      | 输入/输出 | 目标张量，将被原地更新。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16。<br>**不支持广播**：y的形状必须能容纳x的广播结果，即y的任意维度不能为1（除非x对应维度也为1）。<br>不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| x      | 输入      | 源张量，可广播到y的形状。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32、DT_FP16。<br>支持广播：x可以广播到y的形状（如x形状为`[m, 1]`，y形状为`[m, n]`）。<br>不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。                                                 |
| alpha  | 输入      | 缩放因子，用于对x进行缩放。<br>支持的类型为：int、float，默认值为1.0。<br>alpha的数据类型会自动转换为与y一致。                                                                                                            |

## 返回值说明

返回更新后的y张量（与输入y共享同一内存地址），Tensor的数据类型与y相同，Shape与y相同。

## 约束说明

1. **dtype约束**：
   - 相同dtype：支持DT_FP32 + DT_FP32、DT_FP16 + DT_FP16。
   - 混合dtype：仅支持DT_FP32 (y) + DT_FP16 (x)，其他组合不支持。
2. **广播约束**：
   - y张量**不支持广播**。如果y的某个维度为1而x对应维度不为1，将报错。
   - x张量**支持广播**到y的形状。
3. **Shape约束**：y和x的维度数必须相同（1-4维）。
4. **Format约束**：y和x的Format必须一致。
5. **原地更新注意**：由于AXPY是原地操作，y的原始值会被覆盖。如需保留原始y值，请提前clone：

   ```python
   y_backup = pypto.clone(y)  # 备份原始y值
   y.axpy_(x, alpha=2.0)      # y被原地更新
   # 此时y_backup仍保留原始值，可用于后续计算
   ```

6. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape维度应和输出一致。

示例：输入y shape为`[m, n]`，x shape为`[m, n]`（或`[m, 1]`广播场景），输出shape为`[m, n]`，TileShape设置为`[m1, n1]`，则`m1`，`n1`分别用于切分输出的`m`，`n`轴。

```python
pypto.set_vec_tile_shapes(32, 32)
```

### 接口调用示例

#### 基本用法

```python
y = pypto.tensor([1, 3], pypto.DT_FP32)
x = pypto.tensor([1, 3], pypto.DT_FP32)
y.axpy_(x, alpha=2.0)
```

结果示例如下：

```python
输入数据y:   [[1.0 2.0 3.0]]
输入数据x:   [[2.0 3.0 4.0]]
alpha:        2.0
输出数据y:   [[5.0 8.0 11.0]]  # y = 2.0 * x + y
```

#### 广播场景

```python
y = pypto.tensor([64, 64], pypto.DT_FP32)  # y shape: [64, 64]
x = pypto.tensor([64, 1], pypto.DT_FP32)   # x shape: [64, 1] (广播到 [64, 64])
y.axpy_(x, alpha=1.5)
```

#### 保留原始y值

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)
x = pypto.tensor([32, 32], pypto.DT_FP32)

# 如需使用原始y值，提前备份
y_backup = pypto.clone(y)

# 执行AXPY，y被原地更新
y.axpy_(x, alpha=2.0)

# y_backup仍保留原始值，可用于其他计算
diff = pypto.sub(y, y_backup)  # 计算y与原始值的差值
```

#### 混合精度（FP32 + FP16）

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)  # y为FP32
x = pypto.tensor([32, 32], pypto.DT_FP16)  # x为FP16
y.axpy_(x, alpha=1.0)  # 支持FP32(y) + FP16(x)
```

#### 一维场景

```python
y = pypto.tensor([128], pypto.DT_FP32)
x = pypto.tensor([128], pypto.DT_FP32)
pypto.set_vec_tile_shapes(64)
y.axpy_(x, alpha=2.0)
```
