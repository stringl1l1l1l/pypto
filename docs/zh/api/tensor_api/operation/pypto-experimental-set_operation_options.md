# pypto.experimental.set\_operation\_options

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

该接口是编译框架提供的运行时动态配置管理功能的核心部分，它将原本静态地写在配置文件tile\_fwk\_config.json中的参数转变为动态、可编程的指令。

## 函数原型

```python
set_operation_options(*, combine_axis: Optional[bool] = None)
```

## 参数说明

| 参数名               | 输入/输出 | 说明                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| combine_axis         | 输入      | **含义**：在代码生成阶段实现尾轴broadcast inline。<br>**说明**：双目运算(32,1) + (32,128)，不需要将(32,1)先broadcast到(32,128)，而是将(32,1)通过brcb指令扩展到(32,8)，再进行(32,8) + (32,128)。前提是(32,1)必须是连续的。<br>**类型**：bool<br>**取值范围**：{True, False} |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

- 使用场景：尾轴broadcast输入尾轴必须是连续的，否则功能失效。如果前序节点是尾轴reduce，reduce接口能够保证；如果前序节点是COPY_IN，需要在前端保证在gm连续。
- 类型安全：必须确保传入的combine_axis的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit/loop内的编译过程，若未设置，则继承上层jit/loop作用域中的设置。

## 调用示例

```python
pypto.experimental.set_operation_options(combine_axis=True)
```
