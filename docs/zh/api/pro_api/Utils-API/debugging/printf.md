# pypto_pro.language.printf

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

在Kernel运行时按格式串打印标量值或指针，用于调试。格式串遵循C printf语法的受限子集。

该接口通过设备侧打印机制输出，具体查看位置由运行环境的CANN日志配置决定。printf运行在S（标量）流水上，有显著运行时开销，仅用于调试，生产环境应移除。

## 函数原型

```python
pypto_pro.language.printf(
    format_str: str,
    *args,
    loc: bool = False,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| format_str | 输入 | 格式串，必须是编译时常量字符串，不能是运行时变量。<br>支持%d/%i（有符号整数）、%u（无符号整数）、%x（十六进制）、%f（浮点）和%p（指针）。不支持flags、宽度、精度和长度修饰符。<br>不支持的conversion：%%（百分号本身）、%s（字符串）、%c（字符）。 |
| *args | 输入 | 要打印的标量值或Ptr指针，数量和类型须与格式串中的格式说明符匹配。支持Python int、Python float、Python bool以及Kernel标量表达式；使用%p时还支持Ptr指针。<br>%d/%i：接受有符号整数和DT_BOOL，拒绝无符号整数<br>%u：接受无符号整数和DT_BOOL，拒绝有符号整数<br>%x：接受无符号整数，拒绝有符号整数和DT_BOOL<br>%f：仅接受DT_FP32，拒绝其他数据类型<br>%p：仅接受Ptr指针。 |
| loc | 输入 | 可选，是否在输出前打印源文件/行号，取值为True或False（默认）。为True时，在输出前打印调用位置的源文件和行号。 |

## 约束说明

无。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl
# 打印整数
pl.printf("flag=%d, offset=%d\n", flag, offset)

# 打印浮点（仅 FP32）
pl.printf("value=%f\n", value_f32)

# 打印十六进制
pl.printf("addr=0x%x\n", addr_u32)

# 纯文本（无参数）
pl.printf("reached checkpoint A\n")

# 带源码位置
pl.printf("debug: i=%d\n", i, loc=True)
```

示例输出：

| 调用 | 输出（假设flag = 1, offset = 32, value_f32 = 3.14, addr_u32 = 0x1234, i = 5） |
|---|---|
| pypto_pro.language.printf("flag=%d, offset=%d\n", flag, offset) | flag=1, offset=32 |
| pypto_pro.language.printf("value=%f\n", value_f32) | value=3.140000 |
| pypto_pro.language.printf("addr=0x%x\n", addr_u32) | addr=0x1234 |
| pypto_pro.language.printf("reached checkpoint A\n") | reached checkpoint A |
| pypto_pro.language.printf("debug: i=%d\n", i, loc=True) | debug: i=5（loc = True时额外附带源码位置） |
