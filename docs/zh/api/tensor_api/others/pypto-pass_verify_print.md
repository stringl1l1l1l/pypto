# pypto.pass\_verify\_print

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

在精度调试Verify特性使能时，使用该接口保存指定Tensor计算的结果到数据文件。

## 函数原型

```python
pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| *values | 输入      | 含义：指定打印的数据或信息。 <br> 说明：pypto.Tensor：打印tensor的数据；int/pypto.SymbolicScalar：打印相应的值；其他Python对象：打印相应的字符串表示 <br> 类型：List[pypto.Tensor,int,pypto.SymbolicScalar,Object] <br> 取值范围：NA <br> 默认值：NA |
| cond    | 输入      | 含义：指定打印数据满足的条件 <br> 说明：表达式计算结果为1：打印指定数据；表达式计算结果为0：不打印数据；该参数可省略，省略时使用默认值，不支持显式传入None。 <br> 类型：Union[int,pypto.SymbolicScalar] <br> 取值范围：0,1 <br> 默认值：1 |

## 返回值说明

无。

## 约束说明

该函数需在设置pypto.set_verify_options(enable_pass_verify=True)后生效。

## 调用示例

```python
verify_options = {
        "enable_pass_verify": True,
      }

@pypto.frontend.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        pypto.pass_verify_print(t2)
        t3 = pypto.SOME_OP2(t0, t2)
        pypto.pass_verify_print(t3, cond=(idx == 5))
         ...
```
