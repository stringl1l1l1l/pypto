# 算子实现

本目录存放所有 PyPTO IR 算子的**前端注册**：算子名、kwarg schema、类型推导。这些算子的
CCE 产码是分开的，在 `src/interface/pypto_pro/backend/` 下。

## 目录结构

```text
src/ir/op/
├── README.md                    # 本文件
├── README_en.md                 # 本文件的英文版
├── type_inference.cpp           # 下列文件共用的类型推导工具
├── ptr_ops.cpp                  # ptr.*    - 指针与张量视图算子
├── simt_ops.cpp                 # simt.*   - SIMT 算子
├── vf_ops.cpp                   # vf.*     - VF（vector-function）算子
├── debug_ops.cpp                # debug.*  - 调试算子
├── sync_ops/
│   └── sync.cpp                 # system.* - 同步与屏障算子
└── block_ops/
    ├── memory.cpp               # block 作用域查询与 SPR 访问（无 block. 前缀）
    ├── out_memory.cpp           # block.*  - 数据搬运（load、store、move、insert）
    ├── out_elementwise.cpp      # block.*  - 逐元素与 gather 算子
    ├── out_matmul.cpp           # block.*  - matmul 系列
    ├── out_reduction.cpp        # block.*  - 行/列归约与 expand
    ├── sort.cpp                 # block.*  - 排序、归并排序、直方图
    └── struct_ops.cpp           # struct.* - 结构体创建 / 字段写入
```

`out_` 前缀表示**显式输出**约定：目标 tile 是算子的第一个参数，而不是返回值。

## 组织原则

### 按命名空间

每个文件独占一个算子命名空间，命名空间就是算子名的前缀：

| 命名空间    | 文件                              | 算子数 |
|-------------|-----------------------------------|-------:|
| `vf.`       | `vf_ops.cpp`                      |     83 |
| `simt.`     | `simt_ops.cpp`                    |     30 |
| `system.`   | `sync_ops/sync.cpp`               |     21 |
| `block.`    | `block_ops/out_*.cpp`、`sort.cpp` |     66 |
| （顶层）    | `block_ops/memory.cpp`            |     15 |
| `debug.`    | `debug_ops.cpp`                   |      5 |
| `ptr.`      | `ptr_ops.cpp`                     |      3 |
| `struct.`   | `block_ops/struct_ops.cpp`        |      2 |

### 按类别（在 `block_ops/` 内部）

- `out_memory.cpp` —— 张量与 tile 之间的数据搬运
- `out_elementwise.cpp` —— 逐元素算术、比较、选择、gather
- `out_matmul.cpp` —— matmul 及其 accumulate / bias / mx 变体
- `out_reduction.cpp` —— 行列归约，以及配套的 expand
- `sort.cpp` —— 排序与直方图
- `memory.cpp` —— block 作用域查询（`get_block_idx`、`get_block_num`）与 SPR 访问
- `struct_ops.cpp` —— 结构体算子

## 新增一个算子

### 1. 选择或新建类别文件

按算子的命名空间和类别挑文件，或在 `block_ops/` 下新建一个。**新建文件不需要改构建脚本**：
`framework/src/interface/CMakeLists.txt` 用 `file(GLOB ... ir/op/*.cpp ir/op/*/*.cpp)` 收集源文件，
重跑 CMake 即可。

### 2. 用流式 API 注册算子

```cpp
// 示例：在 block_ops/out_matmul.cpp 中
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
namespace {

// 类型推导辅助函数（可选，便于相关算子间复用）
// args 和 kwargs 都会传进来：kwargs 承载算子的属性
TypePtr DeduceBlockMatMulType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << op_name << " requires 3 arguments (out, lhs, rhs)";
    // ... 由 args[0] 的类型推导；当某个属性会改变结果时，再看 kwargs ...
    return args[0]->GetType();
}

} // namespace

REGISTER_OP("block.matmul_example")
    .set_op_category("BlockOp")
    .set_description("Matrix multiplication of two tiles into a pre-allocated output tile")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_attr<bool>("a_trans")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockMatMulType(args, kwargs, "block.matmul_example");
    });

} // namespace ir
} // namespace pypto
```

`REGISTER_OP` 依赖静态初始化，所以库加载时算子会自行注册，没有需要手动调用的注册函数。

关于这几个流式调用，有两点要注意：

- `set_op_category` 用于工具侧分组。取该命名空间已在用的值：
  `BlockOp`、`VFOp`、`SimtOp`、`SyncOp`、`PtrOp`、`StructOp`、`DebugOp`、`LanguageOp`。
- `set_attr<T>("name")` 声明一个关键字参数。它同时是 `ValidateKwargs` 的**白名单**，所以
  一个属性都不声明的算子会**不加拦阻地接受任何关键字**。`add_argument` 只是文档——
  没有任何代码会读回它。

### 3. 实现后端

只做前端注册的算子能被构造，但不能被编译：kernel 用了没有后端的算子，会在 CCE 产码阶段失败，
报 `Unknown call '<op>' reached CCE codegen`。产码实现放在 `src/interface/pypto_pro/backend/` 下，
用 `REGISTER_BACKEND_OP(BackendCCE, "<算子名>")` 注册。

### 4. 编写测试

- C++ 注册契约：`framework/tests/ut/interface/src/ir/`
- Python 侧可见的行为：`python/tests/ut/pypto_pro/ir/op/test_registry.py`
- 后端产出：`framework/tests/ut/interface/src/pypto_pro/backend/`

## 这种结构的好处

1. **模块化**：每个命名空间、每个类别独占一个文件
2. **可维护**：便于定位和修改特定算子
3. **可扩展**：新增算子不会让已有文件膨胀
4. **构建性能**：改动一个类别不会触发其他类别重新编译
5. **组织清晰**：算子该放哪个文件，由它的名字直接决定

## 现有算子

共 225 个算子。下面按文件列出代表性的一批，完整清单请看对应文件。

### 指针与张量视图（`ptr_ops.cpp`）

- `ptr.make_ptr` —— 从张量取出裸指针，或重解释指针的 dtype
- `ptr.make_tensor` —— 基于指针或已有张量构造张量视图
- `ptr.addptr` —— 按元素语义偏移裸指针

### Block 算子

Block 算子面向硬件优化的 block 级编程，作用于 tile，并遵循**显式输出**约定：目标 tile 是第一个参数。

- **block 作用域查询与 SPR**（`block_ops/memory.cpp`，无命名空间前缀）：
  `get_block_idx`、`get_block_num`、`get_subblock_idx`、`get_subblock_num`、`get_spr`、
  `set_saturation_flag`
- **数据搬运**（`block_ops/out_memory.cpp`）：
  `block.load`、`block.store`、`block.move`、`block.move_fp`、`block.insert`、`block.ub_copy`
- **逐元素与 gather**（`block_ops/out_elementwise.cpp`）：
  `block.add`、`block.sub`、`block.mul`、`block.div`（tile-tile）；
  `block.adds`、`block.subs`、`block.muls`、`block.divs`（tile-scalar）；
  `block.cmp`、`block.sel`、`block.gather`、`block.gatherb`、`block.gathermask`
- **Matmul**（`block_ops/out_matmul.cpp`）：
  `block.matmul`、`block.matmul_acc`、`block.matmul_bias`、`block.matmul_mx`、`block.gemv`
- **归约与 expand**（`block_ops/out_reduction.cpp`）：
  `block.row_sum`、`block.row_max`、`block.row_min`、`block.col_sum`、`block.col_max`、
  `block.col_min`、`block.row_reduce`、`block.col_reduce`、`block.row_expand`、
  `block.col_expand`
- **排序**（`block_ops/sort.cpp`）：
  `block.sort32`、`block.mrgsort`、`block.mrgsort2`、`block.histogram`
- **结构体**（`block_ops/struct_ops.cpp`）：
  `struct.create`、`struct.set`

### VF 算子（`vf_ops.cpp`）

vector-function 算子，在 VF section 内发射。寄存器由赋值形式（`dst = vf.xxx(...)`）隐式声明。

- `vf.reg_tensor`、`vf.mask_reg` —— 寄存器声明（不可直接调用）
- `vf.create_mask`、`vf.update_mask` —— 谓词寄存器
- `vf.load_align`、`vf.store_align` —— 对齐的加载 / 存储
- `vf.full`、`vf.add`、`vf.max`、`vf.reduce_sum` —— 广播与计算

### SIMT 算子（`simt_ops.cpp`）

- `simt.thread_idx`、`simt.block_idx`、`simt.block_dim`、`simt.grid_dim`、
  `simt.linear_thread_idx` —— 上下文查询
- `simt.exp`、`simt.sqrt`、`simt.abs`、`simt.fma` —— 标量数学
- `simt.atomic_add`、`simt.atomic_cas` —— 原子操作
- `simt.syncthreads`、`simt.threadfence` —— 同步

### 同步（`sync_ops/sync.cpp`）

- `system.sync_src_dyn`、`system.sync_dst_dyn` —— pipe 事件的 set / wait
- `system.bar_m`、`system.bar_mte1`、`system.bar_mte2`、`system.bar_mte3`、`system.bar_all` —— 屏障
- `system.set_cross_core`、`system.wait_cross_core` —— 跨核同步

### 调试（`debug_ops.cpp`）

- `debug.dump_tensor`、`debug.dump_tile` —— dump 数据
- `debug.printf`、`debug.assert`、`debug.trap`

## 另见

- [类型推导头文件](../../../../include/ir/type_inference.h)
- [类型推导实现](type_inference.cpp)
- [算子注册表头文件](../../../../include/ir/op_registry.h)
- [算子注册表实现](../op_registry.cpp)
- [CCE 后端实现](../../pypto_pro/backend/)
