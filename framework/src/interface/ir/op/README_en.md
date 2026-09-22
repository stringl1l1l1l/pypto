# Operator Implementations

This directory contains the frontend registration for all PyPTO IR operators: each operator's
name, its kwarg schema, and its type deduction. The CCE code generation for these operators
lives separately, under `src/interface/pypto_pro/backend/`.

## Directory Structure

```text
src/ir/op/
├── README.md                    # Chinese version of this file
├── README_en.md                 # This file
├── type_inference.cpp           # Type inference utilities shared by the files below
├── ptr_ops.cpp                  # ptr.*    - pointer and tensor-view operators
├── simt_ops.cpp                 # simt.*   - SIMT operators
├── vf_ops.cpp                   # vf.*     - VF (vector-function) operators
├── debug_ops.cpp                # debug.*  - debug operators
├── sync_ops/
│   └── sync.cpp                 # system.* - synchronisation and barrier operators
└── block_ops/
    ├── memory.cpp               # Block-scope queries and SPR access (no block. prefix)
    ├── out_memory.cpp           # block.*  - data movement (load, store, move, insert)
    ├── out_elementwise.cpp      # block.*  - element-wise and gather operators
    ├── out_matmul.cpp           # block.*  - matmul family
    ├── out_reduction.cpp        # block.*  - row/col reduction and expand
    ├── sort.cpp                 # block.*  - sort, merge-sort, histogram
    └── struct_ops.cpp           # struct.* - struct create / field set
```

The `out_` prefix marks the explicit-output convention: the destination tile is the operator's
first argument rather than its return value.

## Organization Principles

### By namespace

Every file owns one operator namespace, and the namespace is the operator-name prefix:

| Namespace   | File                          | Operators |
|-------------|-------------------------------|----------:|
| `vf.`       | `vf_ops.cpp`                  |        83 |
| `simt.`     | `simt_ops.cpp`                |        30 |
| `system.`   | `sync_ops/sync.cpp`           |        21 |
| `block.`    | `block_ops/out_*.cpp`, `sort.cpp` |    66 |
| (top level) | `block_ops/memory.cpp`        |        15 |
| `debug.`    | `debug_ops.cpp`               |         5 |
| `ptr.`      | `ptr_ops.cpp`                 |         3 |
| `struct.`   | `block_ops/struct_ops.cpp`    |         2 |

### By category (within `block_ops/`)

- `out_memory.cpp` - data movement between tensor and tile
- `out_elementwise.cpp` - element-wise arithmetic, comparison, selection, gather
- `out_matmul.cpp` - matmul and its accumulate / bias / mx variants
- `out_reduction.cpp` - row and column reduce, and the matching expand
- `sort.cpp` - sort and histogram
- `memory.cpp` - block-scope queries (`get_block_idx`, `get_block_num`) and SPR access
- `struct_ops.cpp` - struct operators

## Adding a New Operator

### 1. Choose or create a category file

Pick the file that owns the operator's namespace and category, or add a new file under
`block_ops/`. A new file needs no build change: `framework/src/interface/CMakeLists.txt`
collects sources with `file(GLOB ... ir/op/*.cpp ir/op/*/*.cpp)`, so re-running CMake picks
it up.

### 2. Register the operator using the fluent API

```cpp
// Example: in block_ops/out_matmul.cpp
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
namespace {

// Helper for type deduction (optional, for code reuse across related operators).
// Both args and kwargs are passed in: kwargs carry the operator's attributes.
TypePtr DeduceBlockMatMulType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << op_name << " requires 3 arguments (out, lhs, rhs)";
    // ... deduce from args[0]'s type, and from kwargs when an attribute changes the result ...
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

`REGISTER_OP` uses static initialization, so the operator registers itself when the library
loads; there is no registration function to call.

Two things about the fluent calls:

- `set_op_category` groups the operator for tooling. Use the value the namespace already uses:
  `BlockOp`, `VFOp`, `SimtOp`, `SyncOp`, `PtrOp`, `StructOp`, `DebugOp`, `LanguageOp`.
- `set_attr<T>("name")` declares one keyword argument. It is also the allow-list that
  `ValidateKwargs` checks against, so an operator that declares no attribute accepts any
  keyword without complaint. `add_argument` is documentation only - nothing reads it back.

### 3. Implement the backend

Frontend registration alone makes an operator constructible but not compilable: a kernel that
uses an operator with no backend fails in CCE codegen with `Unknown call '<op>' reached CCE
codegen`. Add the code generation under `src/interface/pypto_pro/backend/` with
`REGISTER_BACKEND_OP(BackendCCE, "<op name>")`.

### 4. Write tests

- C++ registration contract: `framework/tests/ut/interface/src/ir/`
- Python-visible behaviour: `python/tests/ut/pypto_pro/ir/op/test_registry.py`
- Backend output: `framework/tests/ut/interface/src/pypto_pro/backend/`

## Benefits of This Structure

1. **Modularity**: each namespace and category is in its own file
2. **Maintainability**: easy to find and modify a specific operator
3. **Scalability**: adding operators does not bloat existing files
4. **Build performance**: changing one category does not recompile the others
5. **Clear organization**: the file an operator lives in follows from its name

## Current Operators

225 operators in total. A representative sample per file follows; read the file for the
complete list.

### Pointer and tensor views (`ptr_ops.cpp`)

- `ptr.make_ptr` - extract a raw pointer from a tensor, or reinterpret a pointer's dtype
- `ptr.make_tensor` - build a tensor view from a pointer or an existing tensor
- `ptr.addptr` - offset a raw pointer with element semantics

### Block operators

Block operators drive hardware-optimized block-level programming. They work on tiles and
follow the explicit-output convention: the destination tile is the first argument.

- **Block-scope queries and SPR** (`block_ops/memory.cpp`, no namespace prefix):
  `get_block_idx`, `get_block_num`, `get_subblock_idx`, `get_subblock_num`, `get_spr`,
  `set_saturation_flag`
- **Data movement** (`block_ops/out_memory.cpp`):
  `block.load`, `block.store`, `block.move`, `block.move_fp`, `block.insert`, `block.ub_copy`
- **Element-wise and gather** (`block_ops/out_elementwise.cpp`):
  `block.add`, `block.sub`, `block.mul`, `block.div` (tile-tile);
  `block.adds`, `block.subs`, `block.muls`, `block.divs` (tile-scalar);
  `block.cmp`, `block.sel`, `block.gather`, `block.gatherb`, `block.gathermask`
- **Matmul** (`block_ops/out_matmul.cpp`):
  `block.matmul`, `block.matmul_acc`, `block.matmul_bias`, `block.matmul_mx`, `block.gemv`
- **Reduction and expand** (`block_ops/out_reduction.cpp`):
  `block.row_sum`, `block.row_max`, `block.row_min`, `block.col_sum`, `block.col_max`,
  `block.col_min`, `block.row_reduce`, `block.col_reduce`, `block.row_expand`,
  `block.col_expand`
- **Sort** (`block_ops/sort.cpp`):
  `block.sort32`, `block.mrgsort`, `block.mrgsort2`, `block.histogram`
- **Struct** (`block_ops/struct_ops.cpp`):
  `struct.create`, `struct.set`

### VF operators (`vf_ops.cpp`)

Vector-function operators, emitted inside a VF section. Registers are declared implicitly by
the assignment form (`dst = vf.xxx(...)`).

- `vf.reg_tensor`, `vf.mask_reg` - register declarations (not callable directly)
- `vf.create_mask`, `vf.update_mask` - predicate registers
- `vf.load_align`, `vf.store_align` - aligned load / store
- `vf.full`, `vf.add`, `vf.max`, `vf.reduce_sum` - broadcast and compute

### SIMT operators (`simt_ops.cpp`)

- `simt.thread_idx`, `simt.block_idx`, `simt.block_dim`, `simt.grid_dim`,
  `simt.linear_thread_idx` - context queries
- `simt.exp`, `simt.sqrt`, `simt.abs`, `simt.fma` - scalar math
- `simt.atomic_add`, `simt.atomic_cas` - atomics
- `simt.syncthreads`, `simt.threadfence` - synchronisation

### Synchronisation (`sync_ops/sync.cpp`)

- `system.sync_src_dyn`, `system.sync_dst_dyn` - pipe event set / wait
- `system.bar_m`, `system.bar_mte1`, `system.bar_mte2`, `system.bar_mte3`, `system.bar_all` - barriers
- `system.set_cross_core`, `system.wait_cross_core` - cross-core synchronisation

### Debug (`debug_ops.cpp`)

- `debug.dump_tensor`, `debug.dump_tile` - dump data
- `debug.printf`, `debug.assert`, `debug.trap`

## See Also

- [Type Inference Header](../../../../include/ir/type_inference.h)
- [Type Inference Implementation](type_inference.cpp)
- [Operator Registry Header](../../../../include/ir/op_registry.h)
- [Operator Registry Implementation](../op_registry.cpp)
- [CCE Backend Implementations](../../pypto_pro/backend/)
