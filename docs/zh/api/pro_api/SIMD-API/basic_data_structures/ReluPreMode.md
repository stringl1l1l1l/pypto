# pypto_pro.language.ReluPreMode

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

使能随路ReLU操作的枚举，用于L0C Buffer->UB数据搬运场景。

随路ReLU逐元素将负值置零、正值保持不变，计算公式如下：

$$dst = \max(src, 0) = \begin{cases} src & src > 0 \\ 0 & src \leq 0 \end{cases}$$

## 原型定义

```python
PYPTO_DECLARE_ENUM(ReluPreMode,
    NormalRelu      # 使能随路ReLU操作
)
```
