# pypto.loop

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

定义一个循环操作，实现python当中的for循环功能。

## 函数原型

```python
loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]
loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]
```

## 参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| start             | 输入      | 循环的起始值。 |
| stop              | 输入      | 循环的终止值。 |
| step              | 输入      | 每次循环的步长。 |
| **kwargs          | 输入      | - name(str)：循环标识名称，默认生成f"loop_{loop_idx}"。<br> - idx_name(str):  循环索引变量的名称，默认生成f"loop_idx_{loop_idx}"。<br> - submit_before_loop(bool):  是否在循环开始前提交计算，默认为False。开启后会在循环开启前强制提交当前累积的计算任务到AI Core执行。<br> - parallel(bool): 是否将循环标记为可并行调度，默认为False。设置为True时，表示该循环的各个迭代之间不存在任何依赖关系，可以被并行调度执行。需配合 device_sched_parallelism 配置项一起使用。|

## 返回值说明

返回一个生成器，依次生成表示每次迭代值的符号整数。

## 约束说明

loop返回的循环索引变量（如b_idx）为SymInt类型，不支持作为列表下标索引使用。

## 调用示例

```python
for _ in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
   ...
```
