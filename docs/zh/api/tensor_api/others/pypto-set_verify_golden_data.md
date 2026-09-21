# pypto.set\_verify\_golden\_data

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

该接口包括以下必要功能：

- 功能1：将用户执行算子时实际的输入、输出列表设置到工具中，用于工具在粗检时使用同样的输入进行模拟计算
- 功能2：将用户已有的计算基准数据（golden）设置到工具中，用于工具在计算出模拟结果后与基准输出对比，进而粗粒度确定模拟结果的正确性

## 函数原型

```python
set_verify_golden_data(in_out_tensors=None, goldens=None)
```

## 参数说明

| 参数名          | 输入/输出 | 说明                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| in_out_tensors  | 输入      | 含义：将用户执行算子时实际的输入、输出列表按照相同位置对应地设置到检测工具。 <br> 说明：JIT调用模式下，该选项不需设置 <br> 类型：List[Union(pypto.Tensor, torch.Tensor)] <br> 取值范围：NA <br> 默认值：NA |
| goldens         | 输入      | 含义：将用户已有的计算基准数据（golden）输出设置到工具中做对比检测。 <br> 说明：该列表与算子输入、输出参数列表的长度一致、位置对应。其中元素对应算子的输出参数，输入参数位置应设为None。若相应位置设置为None，表示跳过该位置的数据对比。 <br> 类型：List[Union(pypto.Tensor, torch.Tensor)] <br> * 其中torch.Tensor的device属性需为CPU，不支持NPU。 <br> 取值范围：NA <br> 默认值：NA |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

该函数需设置pypto.set_verify_options(enable_pass_verify=True)后生效。

## 调用示例

```python
set_verify_golden_data(goldens=[None, None, golden_out0])
set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```
