# pypto.DYNAMIC

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

`pypto.DYNAMIC`用于定义动态维度（Dynamic Dimension），允许张量的某些维度在运行时变化。这对于处理可变的batch size、序列长度等场景非常有用。动态维度通常在模块级别定义，然后在JIT编译的内核函数的类型注解中使用。

主要应用场景：

- **动态Batch Size**: 推理时batch size可能随请求数量变化
- **动态序列长度**: NLP任务中文本序列长度不固定
- **动态图结构**: 图神经网络中节点数量可变
- **条件计算**: 根据输入形状决定计算流程

## Shape标记方式

| 标记 | 含义 |
| --- | --- |
| `pypto.DYNAMIC`或`pypto.DYN` | 动态轴，传入torch tensor该维变化时**无需重编译** |
| `pypto.STATIC` | 静态轴，传入torch tensor该维变化时**触发重编译** |
| `64` | 固定轴，只允许传入该固定大小，传入其他大小会报错(runtime_debug_mode为3，开启校验) |
| `...` | 剩余轴都作为静态轴处理 |

## 约束说明

1. 动态维度必须在JIT函数的类型注解中使用

## 调用示例

### 示例1: 基础用法 - 动态Batch Size

```python
import pypto

# 固定轴
HIDDEN_SIZE = 128

@pypto.frontend.jit
def add_bias(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    bias: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
):
    # 实现add逻辑
    # [pypto.DYNAMIC, ...]第一维是动态的，省略号表示剩余维度是静态的
    ...

# 可以用不同的batch size调用
x1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
out1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
result1 = add_bias(x1, bias, out1)  # batch=2

x2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
out2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
result2 = add_bias(x2, bias, out2)  # batch=8
```

### 示例2: 多个动态维度

```python
HIDDEN = 768

@pypto.frontend.jit
def attention_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
):
    # 实现attention逻辑
    # 前两维（batch、序列长度）都是动态的
    ...
    return out

# 可以处理不同的batch和序列长度
attention_kernel(q_4_128, k_4_128, v_4_128, out)  # B=4, SEQ=128
attention_kernel(q_2_256, k_2_256, v_2_256, out)  # B=2, SEQ=256，无需重编译
```

## 最佳实践

1. **文档说明**: 在代码注释中说明哪些维度是动态的及其含义
2. **测试覆盖**: 测试不同的动态维度取值，确保代码的正确性
