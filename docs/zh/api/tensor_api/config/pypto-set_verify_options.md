# pypto.set\_verify\_options

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

设置精度调试Verify特性的自检开关、及特性对应的功能选项。

## 函数原型

```python
set_verify_options(*,
                   enable_pass_verify: Optional[bool] = None,
                   pass_verify_save_tensor: Optional[bool] = None,
                   pass_verify_save_tensor_dir: Optional[str] = None,
                   pass_verify_pass_filter: Optional[List[str]] = None,
                   pass_verify_error_tol: Optional[List[float]] = None,
                   ) -> None
```

## 参数说明

| 参数名                          | 输入/输出 | 说明                                                                 |
|---------------------------------|-----------|----------------------------------------------------------------------|
| enable_pass_verify              | 输入      | 含义：总体使能开关，决定所有 *pass_verify_* 选项、接口是否有效。 <br> 说明：True：代表使能。 <br> 类型：bool <br> 取值范围：True/False <br> 默认值：False |
| pass_verify_save_tensor         | 输入      | 含义：配置是否将模拟计算数据存盘。 <br> 说明：True：代表存盘。 <br> 类型：bool <br> 取值范围：True/False <br> 默认值：False |
| pass_verify_save_tensor_dir     | 输入      | 含义：配置检测结果及数据的保存路径。 <br> 说明：设定绝对路径的字符串。 <br> 类型：str <br> 默认值：<br> {RUNNING_DIR}/output/output_{TS} |
| pass_verify_pass_filter         | 输入      | 含义：配置待自检的Pass名称列表。 <br> 说明：合法的Pass名称。 <br> 不指定则默认校验pass: ["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"]；指定"all"则校验所有pass，指定[]不校验pass只校验tensor_graph;指定非法名称则忽略。 <br> 类型：List[str] <br> 默认值：空 |
| pass_verify_error_tol           | 输入      | 含义：配置精度工具对比精度需要用到的rtol和atol。 <br> 说明：List中的第一个值是rtol，第二个值为atol；List长度不等于2时，使用默认值。 <br> 类型：List[float] <br> 默认值：[1e-3, 1e-3] |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

## 调用示例

```python
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```
