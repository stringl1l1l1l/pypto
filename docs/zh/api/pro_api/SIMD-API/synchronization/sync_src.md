# pypto_pro.language.system.sync_src

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

用于同一AI Core内不同流水之间的同步。当set_pipe指定流水中位于本接口之前的所有数据读写操作完成后，pypto_pro.language.system.sync_src将对应标志位置为1。该接口不会阻塞set_pipe中位于其后的操作，需要与[pypto_pro.language.system.sync_dst](sync_dst.md)配对使用。

## 函数原型

```python
pypto_pro.language.system.sync_src(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: Union[int, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| set_pipe | 输入 | pypto_pro.language.PipeType枚举值，表示源流水，即发送同步事件的流水。必须指定一条具体流水，不支持pypto_pro.language.PipeType.ALL。支持的流水组合见约束说明。 |
| wait_pipe | 输入 | pypto_pro.language.PipeType枚举值，表示目的流水，即等待同步事件的流水。必须指定一条与set_pipe不同的具体流水，不支持pypto_pro.language.PipeType.ALL。支持的流水组合见约束说明。 |
| event_id | 输入 | 事件ID。支持Python int、结果为整数的常量表达式或整数类型的运行时Scalar，不支持bool，取值范围为[0, 7]。 |

## 约束说明

- 必须与pypto_pro.language.system.sync_dst配对使用，两个接口的set_pipe、wait_pipe和event_id必须完全一致。
- 必须先调用pypto_pro.language.system.sync_src发送事件，再调用pypto_pro.language.system.sync_dst等待事件。
- set_pipe和wait_pipe必须组成当前执行侧支持的核内同步路径。
- 对于同一set_pipe、wait_pipe组合，同一event_id只能在前一次pypto_pro.language.system.sync_dst完成等待后复用。连续调用pypto_pro.language.system.sync_src、漏写任一配对接口或两个接口所在的控制流路径不一致，可能造成数据竞争或死锁。
- 与auto_mutex=True同时使用时，应确保显式同步与自动mutex分别处理明确的依赖，避免对同一依赖重复同步。

### Vector段支持的流水组合

| set_pipe | 支持的wait_pipe |
|---|---|
| MTE2 | V、MTE3、S |
| MTE3 | V、MTE2、S |
| V | MTE2、MTE3、S |
| S | V、MTE2、MTE3 |

### Cube段支持的流水组合

| set_pipe | 支持的wait_pipe |
|---|---|
| MTE1 | MTE2、M、MTE3、FIX |
| MTE2 | MTE1、M、MTE3、S、FIX |
| MTE3 | MTE1、MTE2、S |
| M | MTE1、MTE2、FIX、S |
| S | MTE2、MTE3 |
| FIX | M、MTE2、S、MTE3、MTE1 |

### 典型同步模式

| 场景 | set_pipe | wait_pipe | 说明 |
|---|---|---|---|
| load后进行向量计算 | MTE2 | V | GM中的数据搬入UB或L1 Buffer后再进行向量计算 |
| 向量计算后store | V | MTE3 | 向量计算完成后再将数据搬出到GM |
| store后再次load | MTE3 | MTE2 | 数据搬出到GM后再向相同Buffer搬入新数据 |

## 返回值说明

无。

## 调用示例

本示例先将两个FP32输入Tensor从GM搬入UB，再调用pypto_pro.language.system.sync_src发送MTE2到V的同步事件。与其配对的pypto_pro.language.system.sync_dst等待该事件后执行逐元素加法。计算完成后，以相同方式同步V和MTE3，最后将结果从UB写入GM。

```python
import pypto_pro.language as pl


@pl.jit()
def sync_src_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_b = pl.make_tile(tt, addr=0x4000)
    tile_out = pl.make_tile(tt, addr=0x8000)
    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])
```
