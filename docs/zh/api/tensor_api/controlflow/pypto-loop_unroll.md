# pypto.loop\_unroll

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

pypto.loop\_unroll是一个支持循环展开的循环迭代器函数，功能与pypto.loop类似，增加了unroll\_list参数支持多个展开方式。

## 函数原型

```python
loop_unroll(start: SymInt = 0, stop: SymInt, step: SymInt = 1, *, name: str = None, idx_name: str = None, unroll_list: List[int] = None, submit_before_loop: bool = False) -> Iterator[Tuple[SymInt, int]]
```

## 参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| *args             | 输入      | 三个可选参数，分别为循环起始值（start），循环结束值（stop），循环步长（step），有以下三种写法：<br> - 单参数形式：stop(SymInt)，起始值默认为0，步长默认为1。等价于：loop_unroll(0, stop, 1)<br> - 双参数形式：start(SymInt)，stop(SymInt)，等价于loop_unroll(start, stop, 1)<br> - 三参数形式：start (SymInt)，stop(SymInt)，step(SymInt)，等价于loop_unroll(start, stop, step) |
| **kwargs          | 输入      | - name(str)：循环标识名称，默认生成f"loop_{loop_idx}"。<br> - idx_name(str):  循环索引变量的名称，默认生成f"loop_idx_{loop_idx}"。<br> - unroll_list(List[int]):  需要展开的循环因子集合，默认为空集合。loop会提供unroll_list中定义的几种展开方式，展开次数为n时，循环步长会变成step*n，每次迭代会执行n次循环体。每种展开次数会生成不同的代码路径。<br> - submit_before_loop(bool):  是否在循环开始前提交计算，默认为False。开启后会在循环开启前强制提交当前累积的计算任务到AI Core执行。 |

## 返回值说明

返回一个迭代器，每次迭代产生一个元组（idx, unroll\_factor\)，idx表示当前循环的索引值，unroll\_factor标识当前选择的展开方式。

## 约束说明

- 展开因子列表会被排序并去重，且总是包含1
- 展开因子按从大到小排序
- 每个展开因子会生成一个子循环
- 多层循环使用了loop_unroll并且配置unroll_list，会大大增加编译出的图数量，影响编译性能

## 调用示例

```python
for idx, unroll_factor in pypto.loop_unroll(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx", unroll_list=[1, 2, 4]):
   ...
```
