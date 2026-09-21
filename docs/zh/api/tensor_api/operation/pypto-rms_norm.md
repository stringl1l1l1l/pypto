# pypto.rms\_norm

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

沿最后一个维度进行根均方层归一化（Root Mean Square LayerNorm，RMSNorm）。如果提供了gamma，则在最后一个维度上应用逐元素缩放。

## 函数原型

```python
rms_norm(input: Tensor, gamma: Tensor = None, epsilon: float = 1e-6) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：PyPTO支持的数据类型。<br>可以是任意Shape的Tensor[..., C]，最后一个维度C通常表示通道数或特征数。 |
| gamma   | 输入      | 可选的缩放参数，Shape应为 [C]。 |
| epsilon | 输入      | 数值稳定性常数，默认值为1e-6。 |

## 返回值说明

归一化后的Tensor，Shape与输入Tensor input相同，输出Tensor将被转换回输入Tensor的原始数据类型。

## 调用示例

```python
x = pypto.tensor([2, 4], pypto.DT_FP32)
gamma = pypto.tensor([4], pypto.DT_FP32)
y = pypto.rms_norm(x, gamma)
```

结果示例如下：

```python
输入数据x: [[1, 2, 3, 4],
            [5, 6, 7, 8]]
输入数据gamma: [1, 1, 1, 1]
输出数据y: [[0.3651, 0.7302, 1.0954, 1.4605],
            [0.7580, 0.9097, 1.0613, 1.2129]]
```
