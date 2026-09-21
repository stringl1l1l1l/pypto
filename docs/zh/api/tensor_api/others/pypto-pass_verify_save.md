# pypto.pass\_verify\_save

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

在精度调试Verify特性使能时，使用该接口保存指定Tensor待计算的结果到数据文件。

## 函数原型

```python
pass_verify_save(
    tensor: Tensor,
    fname: Union[str, SymbolicScalar, int],
    cond: Union[int, SymbolicScalar] = 1,
    **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar]
) -> None
```

## 参数说明

| 参数名   | 输入/输出 | 说明                                                                 |
|----------|-----------|----------------------------------------------------------------------|
| tensor   | 输入      | 含义：pypto kernel function中的pypto.Tensor。 <br> 说明：待保存的Tensor对象。 <br> 类型：pypto.Tensor <br> 取值范围：NA |
| fname    | 输入      | 含义：文件名模板，定义tensor保存的文件名前缀，tensor的内存转储保存至{fname}.data、tensor的元数据（shape,dtype）保存至{fname}.csv。保存路径为：${work_path}/output/output_*/tensor/ <br> 说明：str：简单文件名前缀；包含"$NAME"的待匹配字符串：将$NAME替换为kwargs中NAME对应的值，然后以替换后的字符串作为文件名前缀。 <br> 类型：Union[str, SymbolicScalar, int] <br> 取值范围：NA |
| cond     | 输入      | 含义：指定打印数据的满足条件 <br> 说明：表达式计算结果为1：打印指定数据；表达式计算结果为0：不打印数据；该参数可省略，省略时使用默认值，不支持显式传入None。 <br> 类型：Union[int,pypto.SymbolicScalar] <br> 取值范围：0,1 <br> 默认值：1 |
| **kwargs | 输入      | 指定fname参数中待匹配字符串的值。 |

## 返回值说明

无。

## 约束说明

该函数需设置pypto.set_verify_options(enable_pass_verify=True)或通过@pypto.frontend.jit(verify_options={"enable_pass_verify": True})开启后生效。

## 调用示例

```python
verify_options = {
        "enable_pass_verify": True
      }

@pypto.frontend.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        # 保存t2到文件t2.*
        pypto.pass_verify_save(t2, 't2-fileprefix')
        t3 = pypto.SOME_OP2(t0, t2)
        # 当idx==5时，保存t3到文件t3_debug_loop_5.* 中
        pypto.pass_verify_save(t3, "t3_debug_loop_$idx", cond=(idx == 5), idx=5)
         ...

```
