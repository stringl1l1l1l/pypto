# pypto.SymbolicScalar.max

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

计算两个符号标量的最大值。

## 使用场景

当需要对SymbolicScalar（如从`tensor.shape`获取的动态维度值）进行比较时，应使用此方法。

**与pypto.maximum的区别**：

- `pypto.maximum`：用于Tensor的逐元素最大值计算
- `SymbolicScalar.max`：用于符号标量的最大值计算

**不支持Python三元表达式**：

```python
# ❌ 错误：不支持三元表达式
cur_seq = kv_act_seqs[b_idx]
tmp = cur_seq - s2_idx * s2_tile
actual = tmp if tmp > threshold else threshold

# ✅ 正确：使用.max()方法
actual = (cur_seq - s2_idx * s2_tile).max(threshold)
```

## 函数原型

```python
max(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 待比较的符号标量或整数。 |

## 返回值说明

返回两个值中的最大值。若均为具体值则返回常量，否则返回SymbolicScalar类型的符号表达式。

## 约束说明

- 如果两个值都是具体的，将返回具体的常量值
- 如果至少有一个不是具体的，将返回符号表达式

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.max(s2)
out2 = s1.max(13)
s3 = pypto.SymbolicScalar("x")
out3 = s3.max(2)
```

结果示例如下：

```python
输出数据out1: 10
输出数据out2: 13
输出数据out3: RUNTIME_Max(x, 2)
```
