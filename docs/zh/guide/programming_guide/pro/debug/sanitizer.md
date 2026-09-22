# 内存检测

## 功能说明

Kernel中的内存访问错误（GM越界、tile越界、片上地址重叠、mutex配对错误）通常不会立即导致Host侧异常，而是表现为结果错乱、偶发错误或Kernel挂死，人工排查成本高。内存检测工具（sanitizer）通过在@pypto_pro.language.jit()中配置sanitizer=True开启：框架自动在GM/tile访问点和mutex调用点插桩，每次Kernel启动结束后统一检测并给出结论。

## 使用约束

- 仅供调试使用：插桩与检测会改变Kernel执行开销，性能测量前应在@pypto_pro.language.jit()中移除sanitizer=True。
- 每次启动Kernel会在GM上分配约1GB的日志缓冲，同一位置的重复缺陷会聚合为一条并计数。
- pypto_pro.language.Ptr只携带地址、不携带所指内存的大小，以指针为源的视图无法获知源大小，不做声明shape超出源的检查。
- 检测覆盖Kernel主体内的访问与声明；SIMT线程块内的所有访问（GM/tile/mutex）均不检测。

## 操作步骤

### 开启内存检测

在核函数装饰器上设置`@pypto_pro.language.jit(sanitizer=True)`：

```python
import pypto_pro.language as pl

@pl.jit(sanitizer=True, auto_mutex=True)
def kernel(x: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

sanitizer默认为False，不开启时，Kernel的编译和执行行为完全不变。开启后，每次启动Kernel，检测自动执行，无问题则静默通过；检出问题时，结果以追加模式写入该kernel编译产物目录下的`pypto_sanitizer_report.txt`（与`sanitizer_report.bin`同目录，路径形如`./build/<kernel_name>__<arch>/tk_none/`），同一kernel多次启动的结果汇总在同一文件。

### 检测报告说明

报告文件中每段结果以kernel标签行开头，括号内为该次启动所用kernel的编译产物目录，如`(./build/view_over_source_kernel__a5__d0/tk_none)`。同名kernel存在多个编译实例（不同TilingKey、数据类型或设备）时，该目录用于区分检测结果来自哪个实例；该目录下同时有检测结果文件`pypto_sanitizer_report.txt`和原始记录`sanitizer_report.bin`。

每段结果先给出检出数量，再逐条列出检出项，以GM访问越界为例：

```text
===== kernel 'oob_kernel' (./build/oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

每个检出项由类型、消息、源码位置和Hint组成：消息描述违规事实（含访问参数与逐维越界判定），源码位置为检出项对应的源码文件与行号，可直接定位问题语句，Hint为该类问题的修复提示。同一位置的重复缺陷（如循环中每轮迭代都越界）聚合为一条，并以`same defect hit N times at this location`标注次数。

各检测类型及其级别如下，其中ERROR为确定性错误，必须解决；WARNING为可疑问题，需人工确认是否需要处理：

| 检测类型 | 级别 | 检测内容 |
|---|---|---|
| GM_OUT_OF_BOUNDS | ERROR | GM访问越界：offsets + tile的valid_shape超出tensor shape（最多5维，含getval/setval标量访问、make_tensor声明的shape超出源tensor）。 |
| TILE_OUT_OF_BOUNDS | ERROR | tile访问越界：offset + valid_shape超出TileType声明维度（move/insert、set_validshape、tile标量访问）。 |
| TILE_OVERLAP | WARNING | 两个tile在同一内存空间的地址区间重叠。 |
| MUTEX_UNLOCK_BEFORE_LOCK / UNPAIRED_MUTEX_LOCK | ERROR | mutex_lock/mutex_unlock未按相同(region, mutex_id, pipe)配对。 |

### 关闭内存检测

问题修复后，在@pypto_pro.language.jit()中移除sanitizer=True并恢复生产配置，覆盖全部目标Shape、数据类型、TilingKey和block_dim重新回归，确认问题消失且性能恢复正常。

## 检测示例

### GM访问越界

GM访问越界包含三类：块访问越界、标量访问越界、make_tensor声明shape超出源tensor。

#### 块访问越界

示例说明：行偏移32加tile有效窗口64超出tensor行数64。

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def oob_kernel(x: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16], z: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16]):
    tt = pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Vec)
    a_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pypto_pro.language.section_vector():
        a = a_group.current()
        # 行偏移32：32 + 64 = 96 > 64
        pypto_pro.language.load(a, x, [32, 0])
        pypto_pro.language.store(z, a, [0, 0])
```

```text
===== kernel 'oob_kernel' (./build/oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tensor shape (64, 64) — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:68
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
```

字段解读：

- offsets：本次访问在tensor各维上的起始位置（如load的第三个参数）。
- valid_shape：访问所使用的tile有效窗口。

结论：demo.py第68行的load以偏移[32, 0]访问[64, 64]的tensor，行方向32 + 64 = 96超出64共32个元素，可定位为循环偏移未按tensor行数封顶，需要检查循环边界或偏移计算。

#### 标量访问越界

示例说明：setval的线性偏移64超出tensor的64个元素。

```python
@pypto_pro.language.jit(sanitizer=True, auto_mutex=True)
def gm_scalar_oob_kernel(out: pypto_pro.language.Tensor[[64], pypto_pro.language.DT_INT32]):
    with pypto_pro.language.section_vector():
        # 线性偏移64：超出64元素
        pypto_pro.language.setval(out, 64, 1)
```

```text
===== kernel 'gm_scalar_oob_kernel' (./build/gm_scalar_oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: scalar access at linear offset 64 exceeds 64 elements at demo.py:17
      Hint: getval/setval offsets must stay inside the tensor element count
```

字段解读：

- linear offset：tensor上的元素序号。

结论：demo.py第17行的setval在64个元素的tensor上写入序号64（最后一个合法序号为63），可定位为标量写入位置越界，需要检查该偏移的上界。

#### make_tensor声明shape超出源tensor

示例说明：make_tensor在[128, 128]的源tensor上声明[200, 200]的视图，共享存储导致声明超出源的容量，声明时即检出。

```python
# 源x为[128, 128]，声明[200, 200]
t2 = pypto_pro.language.make_tensor(x, [200, 200])
```

```text
===== kernel 'view_over_source_kernel' (./build/view_over_source_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] GM_OUT_OF_BOUNDS: make_tensor view [200, 200] covers 80000 bytes, source tensor [128, 128] holds 32768 bytes at demo.py:1255
      Hint: A view shares its source's storage; keep its shape/stride size within the source
```

字段解读：

- make_tensor view后的方括号：声明shape。
- covers：按shape和stride展开后实际覆盖的字节数。
- holds：源tensor的字节容量。

结论：demo.py第1255行的视图声明[200, 200]（覆盖80000字节）挂在[128, 128]（32768字节）的源tensor上，可定位为声明超出源的容量，需要核对视图shape与源tensor的实际大小。

### tile访问越界

tile访问越界包含四类：move偏移越界、insert偏移越界、动态set_validshape窗口越界、标量访问越界。

#### move偏移越界

示例说明：move以offset [32, 0]访问声明为[64, 64]的tile，offset加窗口超出声明维度。

```python
# tile声明为[64, 64]，offset 32 + 窗口64 = 96 > 64
pypto_pro.language.move(a_l0a, cur_a, offset=[32, 0])
```

```text
===== kernel 'tile_move_offset_oob_kernel' (./build/tile_move_offset_oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [32, 0] + valid_shape [64, 64] exceed tile shape [64, 64] — dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32) at demo.py:227
      Hint: Keep the tile's valid shape within the declared TileType dims
```

字段解读：

- exceed tile shape后的方括号：TileType声明的tile shape（GM访问越界中对应位置为tensor shape）。

结论：demo.py第227行的move以偏移[32, 0]访问声明为[64, 64]的tile，行方向32 + 64 = 96超出64共32个元素，可定位为tile内偏移超出声明维度，需要检查搬运偏移或改用更大的tile。

#### insert偏移越界

示例说明：insert将[64, 64]的src写入声明为[128, 64]的dst的offset [96, 0]处，写入偏移加窗口超出声明维度。

```python
# dst声明为[128, 64]，src窗口[64, 64]，写入偏移96：96 + 64 = 160 > 128
pypto_pro.language.insert(dst, src, offset=[96, 0])
```

```text
===== kernel 'insert_oob_kernel' (./build/insert_oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: offsets [96, 0] + valid_shape [64, 64] exceed tile shape [128, 64] — dim0: offset 96 + valid_shape 64 = 160 > 128 (over by 32) at demo.py:64
      Hint: Keep the tile's valid shape within the declared TileType dims
```

字段解读：

- exceed tile shape后的方括号：目的tile（dst）声明维度。
- offsets与valid_shape：写入偏移与src的有效窗口，二者之和不得超过dst声明维度。

结论：demo.py第64行的insert将[64, 64]的src写入[128, 64]的dst偏移[96, 0]处，行方向96 + 64 = 160超出128共32个元素，可定位为目的tile声明维度不足或写入偏移过大。

#### 动态set_validshape窗口越界

示例说明：set_validshape设置的动态窗口m运行时为65，超出tile声明维度64，窗口设置语句本身报TILE_OUT_OF_BOUNDS，后续按超限窗口的访问还会连带报GM_OUT_OF_BOUNDS。

```python
# tile声明为[64, 64]，动态窗口m运行时为65（未用min封顶）
pypto_pro.language.set_validshape(t, [m, 64])
pypto_pro.language.load(t, a, [0, 0])
pypto_pro.language.store(out, t, [0, 0])
```

```text
===== kernel 'validshape_oob_kernel' (./build/validshape_oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 3 issue(s) detected
[1/3] GM_OUT_OF_BOUNDS: offsets [0, 0] + valid_shape [65, 64] exceed tensor shape (64, 64) — dim0: offset 0 + valid_shape 65 = 65 > 64 (over by 1) at demo.py:46
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
[2/3] GM_OUT_OF_BOUNDS: offsets [0, 0] + valid_shape [65, 64] exceed tensor shape (64, 64) — dim0: offset 0 + valid_shape 65 = 65 > 64 (over by 1) at demo.py:47
      Hint: The tile valid_shape plus the loop offsets must stay inside the tensor logical shape on every dimension
[3/3] TILE_OUT_OF_BOUNDS: offsets [0, 0] + valid_shape [65, 64] exceed tile shape [64, 64] — dim0: offset 0 + valid_shape 65 = 65 > 64 (over by 1) at demo.py:45
      Hint: Keep the tile's valid shape within the declared TileType dims
```

字段解读：

- [3/3]：窗口设置语句本身的越界（demo.py第45行set_validshape的窗口[65, 64]超出tile声明[64, 64]）。
- [1/3]和[2/3]：后续load/store按超限窗口访问tensor的连带报错。

结论：动态窗口未按tile声明维度封顶，需要用min等手段限制窗口上界。

#### 标量访问越界

示例说明：getval的线性偏移4096超出[64, 64] tile的4096个元素。

```python
# tile声明为[64, 64]（4096元素），线性偏移4096：超出
pypto_pro.language.setval(out, 0, pypto_pro.language.getval(t, 4096))
```

```text
===== kernel 'tile_scalar_oob_kernel' (./build/tile_scalar_oob_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OUT_OF_BOUNDS: scalar tile access at linear offset 4096 exceeds 4096 elements (dims [64,64]) at demo.py:31
      Hint: getval/setval offsets must stay inside the tile element count
```

字段解读：

- dims后的方括号：tile声明维度。

结论：demo.py第31行的getval在[64, 64]（4096个元素）的tile上读取序号4096（最后一个合法序号为4095），可定位为tile标量访问越界，需要检查该偏移的计算。

### tile地址重叠

示例说明：两个tile在UB上的地址区间相交，给出WARNING（可能是刻意的复用安排，也可能是地址规划错误）。

```python
ta_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tb_group = pypto_pro.language.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])  # 与ta_group地址相交
ta = ta_group.current()
tb = tb_group.current()
```

```text
===== kernel 'overlap_kernel' (./build/overlap_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] TILE_OVERLAP: tile range [Vec 0x0, 0x2000) overlaps another tile [Vec 0x1000, 0x3000) at demo.py:124
      Hint: Two tiles must not share on-chip address space
```

字段解读：

- tile range后的方括号：检出项所在tile的内存空间与地址区间（左闭右开）。
- overlaps another tile后的方括号：与之相交的另一个tile的区间。
- 同一内存空间：二者均落在Vec（UB）。

结论：demo.py第124行的两个tile在UB上分别占用[0x0, 0x2000)和[0x1000, 0x3000)，从0x1000起重叠0x1000字节，需要核对两个tile的addr规划或确认是否为有意的复用。

### mutex配对错误

示例说明：mutex_unlock未与相同(region, mutex_id, pipe)的mutex_lock配对。

```python
# 只解锁不上锁
pypto_pro.language.system.mutex_unlock(mutex_id=7, pipe=pypto_pro.language.PipeType.MTE2)
```

```text
===== kernel 'mutex_unlock_before_lock_kernel' (./build/mutex_unlock_before_lock_kernel__a5__d0/tk_none) =====
PyPTO Sanitizer: 1 issue(s) detected
[1/1] MUTEX_UNLOCK_BEFORE_LOCK: mutex_unlock(mutex_id=7, pipe=MTE2, region=1) without a preceding mutex_lock at demo.py:705
      Hint: Every mutex_unlock must be paired with a mutex_lock on the same (region, mutex_id, pipe)
```

字段解读：

- region：执行核编号（不同执行核的mutex硬件槽位独立，配对按region区分）。
- mutex_id：锁编号。
- pipe：该锁保护的流水线。

结论：demo.py第705行的mutex_unlock在region 1上解锁mutex_id为7、pipe为MTE2的锁，但此前没有对应的mutex_lock，可定位为解锁缺少配对的加锁，需要补齐lock或删除多余的unlock。
