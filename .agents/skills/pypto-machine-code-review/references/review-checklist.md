# Machine 侧 C++ 检视清单

面向 `framework/src/machine/` 下 C++ 改动。命中即记录 `{文件, 行号, 级别, 问题, 建议}`。

级别定义：
- **Blocker**：必须修复才能合入（crash 风险、精度错误、数据损坏）
- **Major**：应修复（潜在隐患、设计缺陷、范围越界）
- **Minor**：建议修复（风格、文档、可维护性）

---

## §1 算法正确性

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 1.1 | CRC/哈希多项式与标准一致 | Blocker | `0xEDB88420` 应为 `0xEDB88320`（IEEE-802.3 CRC32） |
| 1.2 | 注释声称的算法名与实现一致 | Major | 注释写"Standard IEEE-802.3"但多项式写错 |
| 1.3 | 整数运算无溢出 | Blocker | `nelm * BytesOf(dtype)` 当 nelm 很大时溢出 |
| 1.4 | 除法/移位无截断误差 | Major | `BitsOf(t)/8` 对 sub-byte 类型（INT4）截断，与 `BytesOf` 不等价 |
| 1.5 | 边界条件（off-by-one、空输入） | Blocker | `len==0` 时仍解引用、`size()==0` 时除零 |

验证方法：对照标准算法实现或头文件中的定义（如 `framework/include/tilefwk/data_type.h` 的 `BytesOf` vs `DataSizeOf`）。

---

## §2 内存安全

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 2.1 | weak 符号调用前判空 | Blocker | `AdxDataDumpServerUnInit()` 直接调用，未判 `if (!func)` |
| 2.2 | 对齐保证与注释一致 | Major | 注释称"4KB aligned"但用 `std::vector`（只保证 16 字节） |
| 2.3 | 缓冲区大小校验 | Blocker | `memcpy_s` 目标缓冲区小于源、TLV `offset+length > totalSize` |
| 2.4 | 空指针解引用 | Blocker | `reinterpret_cast<T*>(nullptr)` 后直接访问 |
| 2.5 | use-after-free / 悬垂指针 | Blocker | 返回局部变量地址、lambda 捕获引用后异步使用 |
| 2.6 | 堆分配在热路径 | Major | AICPU 热路径 `new`/STL（见 §6 / `perf-rules.md` §2.1） |
| 2.7 | `RelocNullable` 仅限绝对地址域；被 reloc 的值可能是 offset=0 时禁止调用 | Blocker | device 侧运行时（`RuntimeAddrRelocWorkspace` 等）对 live pool slot / freelist 中的 **offset 值**调用 `RelocNullable`：offset=0 是合法地址（workspace 起点），却被 `if (addr != 0)` 当作空指针跳过，槽位保持 0 被 `WsSlotAllocator::Allocate` 原样分配出去（OUTCAST_ADDRESS_NULL / 边界 pool 耗尽，见 !6476） |

验证方法：grep `__attribute__((weak))` 确认所有 weak 符号调用点；检查 `AlignedCopy` 类函数的对齐实现；对每处 device 侧运行时 `RelocNullable` 调用确认值域——绝对地址（0 确实代表未分配，如 host 分支的 `runtimeWorkspace_`）可用 `RelocNullable`，offset / 相对地址（0 是合法的 workspace 起点，如 `RuntimeAddrRestore` 拷入 freelist 的 backup offset）必须用 `Reloc` 无条件平移。

---

## §3 缓存/哈希碰撞

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 3.1 | cache key 是否包含运行时 shape | Blocker | `KernelBundleRegistry::CacheByHash` 用 hashKey 去重，new IR 不同 shape 相同 hash → 静默返回错误 bundle |
| 3.2 | hash 算法是否含动态维度具体值 | Major | `ComputeHash` 只哈希 IR 结构，动态维度保持符号 `-1`，不同 shape 相同 hash |
| 3.3 | AOT cache 复用是否安全 | Blocker | `AOTCodePoolManager::FindEntry` 按 hashKey 命中，若 CF binary 是 shape 相关则用错 code |
| 3.4 | 文档/注释与去重逻辑一致 | Major | header 写"v1: single static-shape snapshot"但 registry 按 hash 去重多 shape |

验证方法：同进程跑两个不同 shape，观察第二次是否全 0 / 精度错误；dump `ComputeHash` 的 hash 字符串对比。

---

## §4 范围纪律

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 4.1 | 无关改动混入功能 PR | Major | bundle PR 混入 HubMix 重构、文档删除、测试删除 |
| 4.2 | 删除已有保护逻辑 | Blocker | 删除 weak 符号判空、删除边界检查 |
| 4.3 | 无注释的魔法数字 | Minor | 硬编码 `4096`、`0xFF` 无解释 |
| 4.4 | 函数名与行为不符 | Major | `AlignedCopy` 不保证对齐 |
| 4.5 | 全局变量无锁说明 | Minor | `g_aicpuSoOverride` 注释称"no locking needed"但未说明约束 |

验证方法：`git diff --stat` 看文件范围是否与 PR 标题匹配；逐 hunk 判断"这行改动是否直接服务于 PR 主题"。

---

## §5 Bundle / 离线打包专项

当 PR 涉及 `framework/src/machine/runtime/bundle/` 时额外检查：

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 5.1 | CRC32 多项式 | Blocker | `0xEDB88420` 应为 `0xEDB88320` |
| 5.2 | BundleHeader/TlvHeader 大小断言 | Major | `static_assert(sizeof(BundleHeader)==64)` 缺失 |
| 5.3 | TLV value 4KB 对齐 | Major | `valueOffset % 4096 != 0` |
| 5.4 | 未知 TLV type 跳过而非报错 | Minor | `default: break;` 静默丢弃 |
| 5.5 | magic number 字节序假设 | Minor | 注释写"little-endian"但未校验平台 |
| 5.6 | version 向前兼容 | Major | `hdr.version > kBundleVersion` 直接拒绝旧版无降级 |
| 5.7 | devProgram 对齐实现 | Major | `AlignedCopy` 用 `std::vector` 不保证 4KB |
| 5.8 | registry hashKey 去重碰撞 | Blocker | 见 §3.1 |
| 5.9 | pack/load target 循环依赖 | Major | pack hook 和 load ABI 在同一 target |
| 5.10 | standalone .so 双 singleton | Major | 与普通 pypto stack 共存时 Meyers singleton 分裂 |

验证方法：读 `kernel_bundle_format.h` 的常量定义、`kernel_bundle_crc32.h` 的多项式、`kernel_bundle_packer.cpp` 的布局计算、`kernel_bundle_registry.cpp` 的去重键。

---

## §6 AICPU 热路径性能

当 PR 涉及 `framework/src/machine/` 下 AICPU Control-Flow / Schedule 热路径（`device_*`、`device_sche*`、`dev_workspace`、`item_pool`、`aot_binary`、`spsc_queue`、CF cache、stitch/task build）时额外检查。

详细原则见 [references/perf-rules.md](perf-rules.md)（**必读**后再下评审结论），至少覆盖 §2.1–§2.5、§2.9–§2.11、§2.10.1、§2.14。

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 6.1 | 热路径无 STL 容器 / 业务 `new` | Blocker | `std::vector`/`std::map`/`new` 在 AICPU 热循环（原则 §2.1） |
| 6.2 | 地址用显式指针类型 | Major | `uint64_t` 长期冒充地址（原则 §2.2） |
| 6.3 | 局部对象优先引用传参 | Minor | 无必要 `&local` 指针传参（原则 §2.3） |
| 6.4 | 热点结构体 trivial 默构 + 布局纪律 | Major | 非平凡默构放大成无用写；热点结构塞冷字段（原则 §2.4） |
| 6.5 | 无整结构 memset | Major | `memset(整个 DeviceTask, 0, sizeof(...))`，应 `InitShell` 只清壳字段（原则 §2.5） |
| 6.6 | 能 Host 判断的 if 不放到 AICPU | Major | 未「以存代算」——encode 置 flag，runtime 读（原则 §2.9） |
| 6.7 | 循环边界与不变量外提 | Major | 循环条件里重复算 `B` / `.size()`、体内重复 `GetSource()` / `IsMixArch()`（原则 §2.10） |
| 6.8 | Host 无新增直连 `rtMemcpy` H2D | Blocker | 必须拷贝则 `NormalizedRtMemcpy` / RELAXED（原则 §2.11） |
| 6.9 | Schedule 路径与 CF 路径一并检查 | Major | 只优化 stitch、忽略 `device_sche*` / `aicore_manager` |
| 6.10 | 耗时 debug 操作加宏隔离 | Major | 仅 Host 仿真的 dump 用 `DEV_IF_NONDEVICE`；device 侧需要的耗时 debug 用 `DEV_IF_DEBUG`；禁止裸调 debug 函数仅靠 `#else` 空实现（原则 §2.14.1） |
| 6.11 | 热循环指针遍历，避开带检查的 `[]` | Major | 热循环 `operator[]` / 每轮 `At`；应 `Data()` / `&At(list,0)` 后下标（原则 §2.10.1） |
| 6.12 | 热循环无 `DEV_INFO` | Major | Init / Stitch / Resolve 热循环用 `DEV_VERBOSE_DEBUG`（原则 §2.14） |
| 6.13 | 函数内 `static` 跨线程缓存安全 | Major | 多 Ctrl / Sche 线程下按线程或按 prog 键控（原则 §4） |

验证方法：读 [references/perf-rules.md](perf-rules.md) 原则逐条对照；grep `std::vector|std::map|new |malloc` 在 `framework/src/machine/device/dynamic/*` 热路径文件中是否出现。

---

## §7 时序 / 重叠流水 / OS 节流

当 PR 涉及 `device_sche*`、`aicore_manager`、`device_ctrl`、`dev_start_args`、`device_launcher`、handshake、超时、ring、early-launch、`KernelArgs` / sharedBuffer、控核拓扑时额外检查。

详细原则见 [references/timing-race-rules.md](timing-race-rules.md)（**必读**后再下评审结论）。

| # | 检查项 | 级别 | 典型问题 |
|---|---|---|---|
| 7.1 | 按读者集合处理共享状态：跨流水读者的字段禁止按轮次原地改；仅 ctrl 串行使用的 DevProg 字段允许写 | Blocker | 把本轮 `nrValidAic`/`scheCpuNum` 写回共享 `DevProg.devArgs`；sche 入口仍读共享拓扑 |
| 7.2 | 回收等最后读者（AICore），且 STOP 可见 → 清槽 → barrier → GOODBYE；fence 不得放进可选分支 | Blocker | sche `Deallocate` 后 ctrl 立刻 restore 同 slot；GOODBYE 早于清 `parallelDevTask` |
| 7.3 | 超时即功能失败且在等其他 AICPU 的路径 ≥ OS 节流窗口；可降级探测保持短超时 | Blocker | 硬失败 spin 仍用 50µs，对端被节流约 50ms 后误报 `ARBIT_FAILED` / `WAIT_CTRL_TIMEOUT` |

验证方法：读 [references/timing-race-rules.md](timing-race-rules.md)；对每处共享 store 列出 ctrl/sche/aicore 读者与 early-launch 重叠窗口；对停核/slot 复用核对 fence 顺序；对新增等待核对失败是硬报错还是降级。
