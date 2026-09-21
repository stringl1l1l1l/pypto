# pypto_pro.language.make_tuple

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

创建编译期命名元组，将多个IR变量按字段名聚合。字段访问在编译期解析为对应的原始值，不生成C++结构体，也不产生运行时开销。
[pypto_pro.language.struct](./struct.md)会生成C++结构体，可用于跨Pipeline传递数据，两者有区别。

## 函数原型

```python
pypto_pro.language.make_tuple(**kwargs: Any) -> Any
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| \*\*kwargs | 输入 | 命名字段，以field=value形式传入，至少指定一个字段。字段名必须为合法标识符。字段值支持Kernel内可解析的值表达式，包括Python标量值、Kernel标量表达式、Tensor、Tile、Ptr以及由这些值组成的Python元组，例如(tile0, tile1)。不支持位置参数，也不支持通过字典展开参数，例如make_tuple(\*\*fields)。 |

## 约束说明

返回的命名元组对象支持通过字段名访问打包的变量。字段访问在编译期被常量折叠回原值，不产生运行时开销。

### 使用场景

- 函数返回多个值时，可以将多个Tile或变量聚合后返回。
- 可以将逻辑相关的变量聚合，并通过字段名访问。
- 可以聚合ping-pong缓冲中的多个Tile，并通过字段名访问。

### 与pypto_pro.language.struct的区别

- 需要跨Pipeline传递数据，例如通过SSBUF通信时，使用pypto_pro.language.struct。
- 仅需在同一Pipeline内聚合变量时，使用pypto_pro.language.make_tuple。

## 返回值说明

返回包含所有命名字段的命名元组对象。

## 调用示例

### 创建命名元组并访问字段

```python
import pypto_pro.language as pl

# 场景 1：标量字段打包 + 字段访问
@pl.jit()
def tuple_scalar_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TScalar", a=11, b=22)
    with pl.section_vector():
        t = pl.make_tuple(first=s.a, second=s.b)
        pl.setval(out, 0, t.first + t.second)
        pl.setval(out, 1, t.second - t.first)
# 场景 2：for 循环内打包 + struct 字段中转
@pl.jit()
def tuple_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("LoopT", v=0, cur=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            acc.cur = i
            t = pl.make_tuple(x=acc.cur, y=acc.cur * 10)
            acc.v = acc.v + t.x + t.y
        pl.setval(out, 0, acc.v)
```
