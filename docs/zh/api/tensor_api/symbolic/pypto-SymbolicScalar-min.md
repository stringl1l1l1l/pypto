# pypto.SymbolicScalar.min

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

计算两个符号标量的最小值。

## 使用场景

当需要对SymbolicScalar（如从`tensor.shape`获取的动态维度值）进行比较时，应使用此方法。

**与pypto.minimum的区别**：

- `pypto.minimum`：用于Tensor的逐元素最小值计算
- `SymbolicScalar.min`：用于符号标量的最小值计算

**不支持Python三元表达式**：

```python
# ❌ 错误：不支持三元表达式
cur_seq = kv_act_seqs[b_idx]
tmp = cur_seq - s2_idx * s2_tile
actual = tmp if tmp < s2_tile else s2_tile

# ✅ 正确：使用.min()方法
actual = (cur_seq - s2_idx * s2_tile).min(s2_tile)
```

## 函数原型

```python
min(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 待比较的符号标量或整数。 |

## 返回值说明

两个值中的最小值。

## 约束说明

- 如果两个值都是具体的，将返回具体的常量值
- 如果至少有一个不是具体的，将返回符号表达式

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.min(s2)
out2 = s1.min(3)
s3 = pypto.SymbolicScalar("x")
out3 = s3.min(2)
```

结果示例如下：

```python
输出数据out1: 5
输出数据out2: 3
输出数据out3: RUNTIME_Min(x, 2)
```
