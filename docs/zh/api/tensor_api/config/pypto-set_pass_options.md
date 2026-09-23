# pypto.set\_pass\_options

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

修改Pass优化参数信息。其主要功能是在编译流程中，针对特定的优化策略和具体的Pass，动态修改其运行时参数配置，从而实现精细化的控制和调试。

## 函数原型

```python
set_pass_options(*,
                     vec_nbuffer_setting: Optional[Dict[str, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[str, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[str, int]] = None,
                     sg_set_scope: Optional[Union[int, Tuple[int, bool, bool]]] = None,
                     sg_set_ooo_scope: Optional[int] = None,
                     experimental: Optional[Dict[str, int]] = None,
                     ooo_sched_mode: Optional[str] = None,
                     sg_set_tunevf_mode: Optional[int] = None,
                     )
```

## 参数说明

| 参数名                  | 输入/输出 | 说明                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting     | 输入      | 含义：合图参数，用于配置相同结构Vector子图的合并数量。 <br> 说明：该参数适用于结构相同的Vector子图合并。<br><br>类型：dict[str, int]。支持两种key格式：<br> ① **函数粒度key**：字符串`"func{magic}_{order}"`或`"DEFAULT"`，可实现不同root function间的精细化配置。`func{magic}_{order}`匹配特定function（funcMagic=magic）的特定同构子图组（hashorder=order）；`DEFAULT`匹配所有未显式指定的子图组。配置后合图信息会直接展示在泳道图的hashOrder-hint字段中（含subGraphCount，代表合并前的子图数量，可根据核心数匹配合并力度）。详情参见下方"函数粒度key配置说明"。<br> ② **语义标签key**：任意非func前缀字符串，按语义标签控制合图粒度（详见下方语义标签key配置说明）。可与函数粒度key共存。<br> 取值：<br> {"DEFAULT": 1}：跳过Vector子图合并 <br> {}（空字典）：自动合并，根据可用Vector核心数自动计算合并粒度<br> {"DEFAULT": N, "func8_0": N2}：手动合并，默认粒度为N，func8_0对应子图组粒度为N2<br> 默认值：{} 空字典 <br> 影响Pass范围： NBufferMerge |
| cube_l1_reuse_setting | 输入 | 含义：合图参数，用于配置重复搬运同一GM（Global Memory）数据的子图合并数量。<br> 说明：该参数适用于含有Cube计算的子图合并，与cube_nbuffer_setting同时配置时，先进行此项合并，再进行cube_nbuffer_setting相关合并。<br><br>类型：dict[str, int]。支持两种key格式：<br> ① **函数粒度key**：字符串`"func{magic}_{order}"`或`"DEFAULT"`，可实现不同root function间的精细化配置。合图信息会直接展示在泳道图的hashOrder-hint字段中（含subGraphCount，代表合并前的子图数量，可根据核心数匹配合并力度）。详情参见下方"函数粒度配置说明"。<br> ② **语义标签key**：按语义标签控制（详见下方语义标签key配置说明）。<br> 取值：<br>{"DEFAULT": 1}：跳过L1Reuse合并 <br> {}（空字典）：自动合并<br> {"DEFAULT": N, "func8_0": N2}：手动合并，默认粒度为N<br> 默认值：{} 空字典 <br> 影响Pass范围：L1CopyInReuseMerge |
| cube_nbuffer_setting    | 输入      | 含义：合图参数，用于配置相同结构AIC子图的合并数量。 <br> 说明：该参数适用于结构相同的AIC子图合并，与cube_l1_reuse_setting同时配置时，先进行cube_l1_reuse_setting相关合并，再进行此项合并。<br><br>类型：dict[str, int]。支持两种key格式：<br> ① **函数粒度key**：字符串`"func{magic}_{order}"`或`"DEFAULT"`，可实现不同root function间的精细化配置。合图信息会直接展示在泳道图的hashOrder-hint字段中（含subGraphCount，代表合并前的子图数量，可根据核心数匹配合并力度）。详情参见下方"函数粒度配置说明"。<br> ② **语义标签key**：按语义标签控制（详见下方语义标签key配置说明）。<br> 取值：<br>{"DEFAULT": 1}：跳过AIC子图合并 <br> {}（空字典）：显式开启自动合并<br> {"DEFAULT": N, "func8_0": N2}：手动合并，默认粒度为N<br>默认值：{"DEFAULT": 1}，即默认跳过AIC子图合并 <br> 影响Pass范围：L1CopyInReuseMerge |
| sg_set_scope            | 输入      | 含义：手动控制子图切分参数。<br> 说明：通过为Operation分配scope，使得相同scope_id（非-1）的相邻Operation强制合并归入同一子图，从而覆盖切分算法的自动划分结果。 <br> 类型：`Tuple[int, bool, bool]`或`int` <br> **tuple格式**：`(scope_id, allow_parallel_merge, allow_cross_scope_merge)`，各字段含义如下： <br> - `scope_id`（int）：scope标识，取值范围 -1~2147483647。相同scope_id的相邻Operation归入同一子图；-1表示不参与scope合并，由切分算法决定子图划分。 <br> - `allow_parallel_merge`（bool）：控制同一scope_id下Operation的合并方式。取值True/False。<br>&emsp;&emsp;False（默认）：仅允许存在上下游连接通路的Operation合并，即Operation A的输出作为Operation B的输入时才可合并到同一子图。<br>&emsp;&emsp;True：允许位于并行分支（无数据依赖）的相同scope_id的Operation也合并到同一子图。 <br> - `allow_cross_scope_merge`（bool）：控制带有scope的子图是否可与无scope（scope_id=-1）的子图合并，扩大scope子图。取值True/False。<br>&emsp;&emsp;False（默认）：带有scope的子图保持独立，不与其他子图合并。<br>&emsp;&emsp;True：允许带有scope的子图与scope_id=-1的子图合并。不同scope_id的子图之间不可合并。 <br> **int格式**：传入单个int时等价于`(scope_id, False, False)`，即仅设置scope_id，不允许并行分支合并和跨scope合并。 <br> 默认值：(-1, False, False) <br> 影响Pass范围：GraphPartition <br> 配置建议：1）视图类Operation与其对应的计算类Operation应配置相同的scope_id。2）Reshape Operation较为特殊，部分场景会单独成子图，手动控制合图行为可能失效。|
| sg_set_ooo_scope            | 输入      | 含义：控制MIX子图内的OoO调度。<br> 说明：通过为 Operation分配ooo_scope，使得相同ooo_scope_id（非-1）的相邻 Operation 强制合并归入同一ooo_task，从而让相同ooo_scope_id（非-1）的相邻Operation在OoO调度生成的流水上尽可能相邻。不允许并行分支合并和跨ooo_scope合并, 不允许不同loop循环次数下的Operation的合并。 <br> 类型：`int`，即设置ooo_scope_id <br> 默认值：-1 <br> 取值范围：-1或1~10000 <br> `ooo_scope_id`：ooo_scope标识。相同ooo_scope_id的相邻Operation归入同一ooo_task；-1表示不参与ooo_scope合并，由切分算法决定ooo_task划分。 <br> 影响Pass范围：OoOSchedule <br> 配置要求：该功能仅对MIX子图生效。同时使用该功能和loop unroll时，unroll设置不得大于10000。该参数计划在后续版本废弃。|
| ooo_sched_mode            | 输入      | 含义：控制MIX子图内的OoO调度。<br> 说明：设置MIX子图内ooo_task的流水调度模式。 <br> 类型：`str` <br> 默认值："" <br> 取值范围：{"", "GAPMIN", "HLF"} <br> 影响Pass范围：OoOSchedule <br> 配置说明：取值为""（默认）时使用基于拓扑序遍历和局部搜索的调度（GapMin调度 + local-search）；取值为"GAPMIN"时仅执行GapMin调度，跳过local-search; 取值为"HLF"时使用Highest Level First调度（按任务到汇点最长路径降序排列后做EFT插入调度）。|
| sg_set_tunevf_mode           | 输入      | 含义：控制VF（Vector Fusion）调优Pass的行为模式。<br> 说明：用于控制TuneTileOpSeqForVF和TuneSyncForVF两个Pass的执行行为。<br> 类型：`int` <br> 默认值：0 <br> 取值范围：{0, 1, 2} <br> - 0：均衡模式，在OoO Pass输出的op序列的基础上自动完成op顺序的调整，自动平衡Pipeline流水与VF融合的整体性能收益。<br> - 1：指令流水优先模式，不改变OoO排好的op执行序。<br> - 2：vf融合优先模式，不考虑性能建模的收益评估，尽量调整op顺序以保证更大范围的VF融合。该模式仅影响TuneSyncForVF中的NeedAdjustOpSeq判断，不影响TuneTileOpSeqForVF。<br> 影响Pass范围：TuneTileOpSeqForVF, TuneSyncForVF |

## 返回值说明

无。

## 约束说明

- 设置时机：不要求在图编译开始前调用，可以在任何时候进行设置。
- 类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit或者loop内的编译过程，若未设置，则继承上层作用域。
- sg_set_scope一致性约束：同一scope_id的所有Operation必须设置相同的`allow_parallel_merge`和`allow_cross_scope_merge`，否则编译报错。
- scope_id为 -1时，`allow_parallel_merge`和`allow_cross_scope_merge`必须为False。
- 不同scope_id的子图之间不可合并，`allow_cross_scope_merge`仅控制带scope的子图与无scope（scope_id=-1）的子图合并。
- sg_set_scope使用说明：
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：支持纯Vector、纯Cube以及CV混合场景的scope配置。
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3系列产品：支持纯Vector或纯Cube的scope配置，不支持CV混合场景的scope配置。
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2系列产品：支持纯Vector或纯Cube的scope配置，不支持CV混合场景的scope配置。
   <!-- end id9 -->
- sg_set_ooo_scope使用说明：
   <!-- npu="950" id10 -->
   - Ascend 950PR&950DT系列产品：支持。
   <!-- end id10 -->
   <!-- npu="A3" id11 -->
   - Atlas A3系列产品：不支持，因为不支持cv mix合图。
   <!-- end id11 -->
   <!-- npu="910b" id12 -->
   - Atlas A2系列产品：不支持，因为不支持cv mix合图。
   <!-- end id12 -->
- ooo_sched_mode使用说明：
   <!-- npu="950" id16 -->
   - Ascend 950PR&950DT系列产品：支持。
   <!-- end id16 -->
   <!-- npu="A3" id17 -->
   - Atlas A3系列产品：不支持，因为不支持cv mix合图。
   <!-- end id17 -->
   <!-- npu="910b" id18 -->
   - Atlas A2系列产品：不支持，因为不支持cv mix合图。
   <!-- end id18 -->
- sg_set_tunevf_mode使用说明：
   <!-- npu="950" id19 -->
   - Ascend 950PR&950DT系列产品：支持。
   <!-- end id19 -->
   <!-- npu="A3" id20 -->
   - Atlas A3系列产品：不支持，因为不支持cv mix合图。
   <!-- end id20 -->
   <!-- npu="910b" id21 -->
   - Atlas A2系列产品：不支持，因为不支持cv mix合图。
   <!-- end id21 -->

## 调用示例

```python
   # 函数粒度配置（func{magic}_{order} 格式）
   pypto.set_pass_options(
       vec_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_l1_reuse_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1})

   # 纯DEFAULT
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2})

   # 语义标签key配置（可与函数粒度key共存）
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("")
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})
```

### dict类型配置说明（函数粒度key / 语义标签key）

### 函数粒度key配置说明（func{magic}\_{order}）

#### 功能概述

通过`"func{magic}_{order}"`格式的key，可以针对**特定function** 的**特定同构子图组**（hashorder）设置合并粒度，实现不同root function间的精细化配置。配置的hashOrder和subGraphCount信息会直接展示在泳道图的hashOrder-hint字段中，格式为`l1ReuseInfo hashOrder: func8_0, subGraphCount: 24`，可根据子图数量和核心数匹配合并力度。

#### 键值对含义

Key: 字符串格式`"func{magic}_{order}"`或`"DEFAULT"`。<br>

- `"func{magic}_{order}"`：匹配functionMagic为magic的function中，hashorder为order的同构子图组。<br>
- `"DEFAULT"`：匹配所有未显式指定的同构子图组。<br>

Value (N): 表示合并粒度。即同构子图组内每N个子图合并为一个新子图执行。N=1表示不合并。

#### 格式约束

- `func`前缀必须小写。
- magic和order必须为整数。
- magic和order之间以下划线`_`分隔。
- 示例合法key：`"func0_0"`、`"func123_5"`、`"func8_1"`。
- 示例非法key：`"Func0_0"`（大写F）、`"func_0"`（缺少magic）、`"func123"`（缺少order）。

#### 配置行为

Pass在处理当前function的子图合并时，遵循"func{magic}\_{order} 精确匹配 > DEFAULT默认配置 > 自动处理"的逻辑：<br>

- 精确匹配：若funcMagic和hashorder双双命中，则按其对应的Value N进行合并。<br>
- DEFAULT默认配置：若未精确命中，但字典中存在`DEFAULT`，则按`DEFAULT`对应的Value执行合并。<br>
- 自动处理：若既未精确命中也无`DEFAULT`，则自动计算合并粒度。<br>

#### 配置示例

| 配置 | 说明 |
|------|------|
|`{"DEFAULT": 1}`|所有同构子图组跳过合并。|
|`{"DEFAULT": 4, "func8_0": 1, "func8_1": 1}`|默认4个同构子图为一组进行合并，func8中hashorder 0和1的子图组跳过合并（不合并）。|
|`{"DEFAULT": 2, "func8_1": 4}`|默认2个同构子图为一组进行合并，func8中hashorder 1的子图四个为一组进行合并。|
|`{"func8_0": 2}`|func8函数中，hashorder为0的子图两个为一组进行合并；其他同构子图组，根据硬件核心数自动计算合并粒度并进行合并。|

### 语义标签key配置说明

#### 功能概述

除函数粒度key外，`vec_nbuffer_setting`、`cube_l1_reuse_setting`和`cube_nbuffer_setting`还支持使用字符串key，即通过`pypto.set_semantic_label`设置的语义标签名称。字符串key允许用户精确控制特定operation所在子图（允许多个）的合并粒度，无需关心其hashorder编号。

#### 字符串键值对含义

Key (label): 语义标签名称，须与至少一个operation的`semantic_label`完全匹配。若未匹配到任何operation，编译时不会报错，仅打印WARN级别的日志并忽略该key。<br>
Value (N): 表示合并粒度。<br>

#### 优先级机制

字符串key的优先级**高于**函数粒度key的默认配置。处理流程为：<br>

1. 首先根据函数粒度key（`func{magic}_{order}` / `DEFAULT`）确定各同构子图组的基础合并粒度。<br>
2. 然后字符串key的值**直接替换**（而非取max）对应子图组的合并粒度。<br>
3. 当多个不同的字符串label指向同一个同构子图组时，取这些label值中的最大值。<br>

#### vec_nbuffer_setting / cube_nbuffer_setting的语义标签行为

字符串key覆盖其所在operation对应的**整个同构子图组**的合并粒度。

#### cube_l1_reuse_setting的语义标签行为

与`vec_nbuffer_setting`和`cube_nbuffer_setting`不同，`cube_l1_reuse_setting`的字符串key **仅作用于包含对应标签operation的子图**，不展开到整个同构组。即同构组内可能只有部分子图被字符串key覆盖，其他子图保持函数粒度key的值。

#### 语义标签配置示例

| 配置                                | 说明                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
|{"DEFAULT": 2, "V1": 1}|所有同构子图组默认合并粒度为2；但V1标签所在的同构子图组合并粒度被替换为1。|
|{"V1": 3}|V1标签所在的同构子图组合并粒度为3；其他同构子图组自动计算合并粒度。|
|{"DEFAULT": 2, "V1": 1, "V2": 3}|默认合并粒度为2；V1所在组替换为1；V2所在组替换为3。若某一组同时有V1和V2两种OP，则取max(1, 3) = 3。|

#### 配置示例

```python
   # 混合函数粒度key和语义标签key配置
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("") # 通过更改语义标签，来精确控制只有该mul OP的语义标签是"V1"
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})

   # 纯语义标签key配置
   pypto.set_pass_options(cube_l1_reuse_setting={"MM1": 4})
```

### sg_set_scope配置说明

#### 配置示例

```python
# int格式：等效于(10, False, False)，仅设置scope_id
pypto.set_pass_options(sg_set_scope=10)

# tuple格式：scope_id=1，允许并行分支合并，不允许跨scope合并
pypto.set_pass_options(sg_set_scope=(1, True, False))

# tuple格式：scope_id=2，允许与无scope的子图合并
pypto.set_pass_options(sg_set_scope=(2, False, True))

# 恢复默认（不参与scope合并，由合图算法自动决定）
pypto.set_pass_options(sg_set_scope=-1)
```

#### 典型场景

<!-- npu="A3" id22 -->
- Atlas A3系列产品：不支持场景二
<!-- end id22 -->
<!-- npu="910b" id23 -->
- Atlas A2系列产品：不支持场景二
<!-- end id23 -->

##### 场景一：整张计算图不切分

当需要将整个计算图保持不切分时，因数据切块会产生多条并行分支，这些分支之间无直接数据依赖，默认会被切分算法拆为独立子图。推荐设置`sg_set_scope=(scope_id, True, False)`，通过`allow_parallel_merge=True`使相同scope_id的并行分支Operation合并到同一子图。

##### 场景二：CV混合场景，构造Mix子图以减少GM搬运

当Cube操作的前后均有Vec操作时，目标是构造一个包含Cube和Vec的Mix子图，避免中间结果在GM上反复搬运。根据是否明确scope边界，分为以下两种情况：

**场景2.1：明确scope边界**

当可以明确划分Cube操作及其紧邻Vec操作的边界时，使用`(scope_id, False, False)`标记边界，使Cube和紧邻的Vec强制归入同一子图，形成Mix子图。

```python
# 明确标记Cube及紧邻Vec为同一scope，形成Mix子图
pypto.set_pass_options(sg_set_scope=(1, False, False))
# ... Cube操作...
# ... 紧邻的Vec操作...
pypto.set_pass_options(sg_set_scope=-1)
```

**场景2.2：不明确scope边界，仅标记CV合并边界**

当无法明确划分边界，但需要Cube前后的Vec子图与Cube子图合并形成Mix子图时，使用`(scope_id, False, True)`标记Cube及其紧邻的Vec作为合并锚点。通过`allow_cross_scope_merge=True`，该scope子图中的Vec可与前后无scope（scope_id=-1）的Vec子图合并，形成更大的子图以减少GM数据搬运，并与Cube子图一起形成Mix子图。

```python
# 前置Vec操作（scope_id=-1，由切分算法自动决定）
vec_out = some_vec_op(x)

# 标记CV合并锚点，允许与无scope的相邻子图合并
pypto.set_pass_options(sg_set_scope=(1, False, True))
# ... Cube操作...
matmul_result = pypto.matmul(vec_out, w)
# ... 紧邻的Vec操作...
pypto.set_pass_options(sg_set_scope=-1)

# 后续Vec操作（scope_id=-1），可与上方scope子图合并为更大子图
result = other_vec_op(add_result)
```

### sg_set_ooo_scope 配置说明

#### 配置示例

```python
# 设置 ooo_scope_id
pypto.set_pass_options(sg_set_ooo_scope=10)

# 恢复默认（不参与 ooo_scope 合并，由切分算法自动决定）
pypto.set_pass_options(sg_set_ooo_scope=-1)
```

#### 典型场景：控制 Operation 的执行顺序

如下面示例，通过将 mul 和 exp 包在相同的 ooo_scope，可以使得 OoO 的调度结果中 exp 在 add 之前执行

```python
# 因为需要在 MIX 子图内使能 ooo_scope，所以使用 sg_set_scope 构造 MIX 子图
pypto.set_pass_options(sg_set_scope=1)
# Cube 操作
matmul_result = pypto.matmul(a, b)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec 操作（ooo_scope_id=1）
c = pypto.mul(matmul_result, scale)
pypto.set_pass_options(sg_set_ooo_scope=-1)
d = pypto.add(c, bias)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec 操作（ooo_scope_id=1）
e = pypto.exp(c)
pypto.set_pass_options(sg_set_ooo_scope=-1)
pypto.set_pass_options(sg_set_scope=-1)
```

### sg_set_tunevf_mode 配置说明

#### 配置示例

```python
# 指令流水优先模式，不改变OoO排好的op执行序
pypto.set_pass_options(sg_set_tunevf_mode=1)

# vf融合优先模式，不考虑性能建模的收益评估，尽量调整op顺序以保证更大范围的VF融合
pypto.set_pass_options(sg_set_tunevf_mode=2)

# 恢复默认行为，即均衡模式
pypto.set_pass_options(sg_set_tunevf_mode=0)
```
