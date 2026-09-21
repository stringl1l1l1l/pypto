# pypto.experimental.assume_divisible

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

声明符号标量表达式在运行时一定可以被指定整数整除。该接口返回表示expr的SymbolicScalar：SymbolicScalar输入原样返回，int输入会转换为SymbolicScalar。同时，接口在Program上注册规范化后的编译优化假设。例如，已知动态长度vm是tile大小的整数倍时，编译器可据此化简逐tile的动态valid_shape，从而使能dualdst等依赖静态valid shape的优化。

## 函数原型

```python
assume_divisible(expr: SymbolicScalar | int, divisor: int) -> SymbolicScalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| expr | 输入 | 要声明整除关系的符号标量表达式或整数常量。 |
| divisor | 输入 | 整除因子，必须为正整数。 |

## 返回值说明

返回表示expr的SymbolicScalar。SymbolicScalar输入原样返回，int输入会转换为SymbolicScalar。

## 约束说明

- 默认情况下，这是无运行时检查的语义契约。用户必须保证运行时始终满足expr % divisor == 0；违约可能导致静默错误结果。
- 开启runtime_debug_mode=4后，编译器会将每个由用户显式登记并规范化后的假设表达式转换为控制流入口处的取模断言；生成代码使用实际运行时值计算该表达式，违约时报错。未通过assume_divisible显式登记的编译器推导表达式不检查。
- 接口只提供优化信息，不改变expr的值，也不会将动态表达式替换为divisor。
- 假设在Program上登记，同一Program中的所有Function均可查询。
- 编译期常量不能整除时会抛出ValueError；可以整除时为平凡no-op。
- divisor <= 0时会抛出ValueError。

## 调用示例

```python
tile_m = 128
valid_m = pypto.experimental.assume_divisible(a.shape[0], tile_m)
tile = pypto.view(mm, [tile_m, 128], [0, 0], valid_shape=[valid_m, 128])
```
