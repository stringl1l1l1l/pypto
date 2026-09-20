# pypto_pro.language.struct_array

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

结构体数组，N个相同的struct按索引存取。

## 函数原型

```python
pypto_pro.language.struct_array(
    size: int,
    *args: Any,
    **kwargs: Any,
) -> Any
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| size | 输入 | 数组长度。<br>- 必须是编译时常量正整数。 |
| \*args | 输入 | 结构体类型名。必须且只能传入一个字符串常量。该名称只能包含字母、数字和下划线，不能以数字开头，且不能是C++关键字。 |
| \*\*kwargs | 输入 | 结构体成员变量，以field=value形式传入，至少指定一个成员变量。不支持通过字典展开参数，例如struct_array(2, "run_info", \*\*fields)。<br>- 成员变量名称不可重复。<br>- 成员变量名称只能包含字母、数字和下划线，不能以数字开头，且不能是C++关键字。<br>- 成员变量仅支持如下类型：<br>&nbsp;&nbsp;- **标量**：初始值支持整数、浮点数、布尔值或Scalar表达式。<br>&nbsp;&nbsp;- **一维标量数组**：一维非空标量数组，数组长度和元素类型必须在编译期确定，元素应为同一数据类型。不支持将嵌套的具名结构体（通过make_tuple、struct创建）作为成员变量。<br>- 标量成员变量的值可通过arr[i].field = value修改，数组成员变量可通过arr[i].field[index]读写，仅支持相同类型的赋值操作。 |

## 约束说明

- 对结构体进行赋值时，必须保证等式两边的结构体类型名，成员变量的名称、顺序、标量类型和数组长度必须完全相同。例如，对同一个变量，在if/else分支中分别对其进行struct赋值，那么这两个struct必须满足上述条件。
- 在控制流（if/else/for/while）中，如果用值拷贝的方式创建struct，得到的是独立副本，修改它不会影响到原始struct。
- 通过下标访问结构体数组元素或一维数组字段元素时，下标必须在对应数组的有效索引范围内。

## 返回值说明

返回一个结构体数组对象。每个元素可通过arr[i].field访问标量字段、arr[i].field[j]访问数组字段元素。

## 调用示例

### 按索引读写字段

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # 创建 2 槽结构体数组：标量字段 + 数组字段
    run_infos = pl.struct_array(2, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    # 按索引修改标量字段
    run_infos[0].batch_id = 7
    run_infos[1].batch_id = 9

    # 按索引修改数组字段元素
    run_infos[0].innerS1Realsize[3] = 128
    run_infos[1].innerS1Realsize[1] = 64

    # 读回数组字段元素并输出
    with pl.section_vector():
        pl.setval(out, 0, run_infos[0].batch_id)
        pl.setval(out, 1, run_infos[1].batch_id)
        pl.setval(out, 2, run_infos[0].innerS1Realsize[3])
        pl.setval(out, 3, run_infos[1].innerS1Realsize[1])
```

### 循环读写数组字段

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_field_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # 创建 4 槽结构体数组，含数组字段
    run_infos = pl.struct_array(4, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    with pl.section_vector():
        # 数组字段元素赋值（arr[i].field[j] = val）
        for i in pl.range(0, 4):
            run_infos[i].innerS1Realsize[0] = i * 100

        # 数组字段元素读取（arr[i].field[j]）
        for i in pl.range(0, 4):
            pl.setval(out, i, run_infos[i].innerS1Realsize[0])
```
