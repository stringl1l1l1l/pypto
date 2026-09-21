---
name: pypto-machine-code-review
description: PyPTO machine 侧（framework/src/machine）代码检视与行级 review 意见提交。覆盖 CANN weak 符号保护、CRC/哈希算法正确性、AOT/cache 碰撞、内存对齐、弱符号空指针、离线打包 bundle、AICPU 热路径高性能编码（禁 STL/堆分配、显式指针、平凡默构、以存代算、循环外提、Host 严禁随意 rtMemcpy）、时序/重叠流水/OS 节流（跨流水共享状态、slot 复用、硬失败等待）等 machine 侧 C++ 改动的检视清单，并通过 GitCode API 将检视意见提交到 PR 的具体代码行（diff_comment）。当用户提到 review PR、检视代码、代码评审、machine code review、行级评论、diff_comment、提交检视意见、AICPU 性能、禁容器、平凡初始化、热路径、ctrl cpu、schedule、时序竞态、OS 节流、early launch、handshake 超时、常稳时使用。触发词：review PR、检视代码、代码评审、machine code review、行级评论、diff_comment、提交检视意见、AICPU 高性能编码、禁容器、平凡初始化、热路径、时序竞态、OS 节流、early launch、handshake 超时、常稳。
---

# PyPTO Machine Code Review

对 `framework/src/machine/` 下 C++ 改动进行代码检视（含 AICPU 热路径性能原则、时序/重叠流水纪律），并将检视意见通过 GitCode API 提交到 PR 的**具体代码行**（diff_comment，显示在 diff 视图对应行旁）。

## 何时使用

- 用户要求 review / 检视某个 PR 的 machine 侧代码
- 用户要求把检视意见"提到代码上"/"提到代码位置"（行级 diff_comment，而非 PR 级普通评论）
- 用户要求审查 `framework/src/machine/` 下 C++ 改动（bundle/launcher/runner/device/aot_binary/device_sche 等）
- 用户修改 / 新增 AICPU Control-Flow / Schedule 热路径代码（`device_*`、`device_sche*`、`dev_workspace`、`item_pool`、`aot_binary`、`spsc_queue`、CF cache、stitch/task build）
- 用户提到 AICPU 性能、禁容器、平凡初始化、热路径、ctrl cpu、schedule、`MakeDynDeviceTask`
- 用户提到时序竞态、OS 节流、early launch、handshake 超时、常稳、控核拓扑、slot 复用

**不要**用本 skill 替代：
- workspace 内存诊断 → `pypto-machine-workspace`
- 精度问题 → `pypto-precision-overall`
- Pass 模块分析 → `pypto-pass-*`

## Agent 工作流

### 阶段 1：获取 PR diff

1. **确定 PR**：用户提供 PR 号（如 `4974`）或分支名。
2. **拉取分支**：
   ```bash
   git fetch https://gitcode.com/<author>/pypto.git <branch> -q
   PR_HEAD=$(git rev-parse FETCH_HEAD)
   ```
3. **确定 base**：`BASE=$(git merge-base FETCH_HEAD upstream/master)`
4. **聚焦 machine 侧**：排除 docs/tests 噪声
   ```bash
   git diff --stat $BASE..$PR_HEAD -- 'framework/src/machine/*' 'framework/include/machine/*'
   ```
5. **逐文件读 diff**：对每个改动文件 `git diff $BASE..$PR_HEAD -- <file>`，定位问题行。

### 阶段 2：对照检视清单

读 [references/review-checklist.md](references/review-checklist.md)（**必读**后再下评审结论），至少覆盖：
- §1 算法正确性（CRC/哈希多项式、整数溢出）
- §2 内存安全（weak 符号空指针、对齐、越界、`RelocNullable` offset-0 判空语义）
- §3 缓存/哈希碰撞（AOT cache、registry 去重键）
- §4 范围纪律（无关改动混入、死代码删除）
- §5 bundle 专项（format/header/TLV/CRC/对齐）
- §6 AICPU 热路径性能（禁 STL/堆分配、显式指针、平凡默构、以存代算、循环外提、指针遍历、Host 严禁随意 rtMemcpy）
- §7 时序 / 重叠流水 / OS 节流（跨流水共享状态、slot 复用与先清后放、硬失败等待覆盖节流）

**若 §6 命中**（PR 涉及 AICPU 热路径），额外读 [references/perf-rules.md](references/perf-rules.md)（详细原则，至少覆盖 §2.1–§2.5、§2.9–§2.11、§2.10.1、§2.14）。

**若 §7 命中**（PR 涉及 `device_sche*` / `aicore_manager` / `device_ctrl` / `dev_start_args` / `device_launcher` / handshake / 超时 / ring / early-launch / sharedBuffer / 控核），额外读 [references/timing-race-rules.md](references/timing-race-rules.md)。

**同时读** [references/memory-rules.md](references/memory-rules.md)（内存规则），将其与内置清单合并执行。该文件表格为空时跳过；非空时按相同流程命中记录。

命中即记录：`{文件, 行号, 级别(Blocker/Major/Minor), 问题描述, 建议}`。

### 阶段 2.5：用户确认（强制）

**提交前必须向用户展示全部检视意见并获确认，禁止跳过。**

1. 以表格形式展示所有命中意见：

   | # | 级别 | 文件:行 | 问题摘要 | 建议修复 |
   |---|---|---|---|---|

2. 使用 `question` 工具询问用户，提供以下选项：
   - **全部提交**：提交全部意见到 PR 代码行
   - **选择性提交**：用户指定提交哪些（按编号）
   - **修改后提交**：用户指出需修改的意见，Agent 修改后重新确认
   - **不提交**：仅输出报告，不提交到 GitCode

3. 用户确认前**禁止调用** `submit_review_comments.py` 或 curl POST。GitCode API 不支持 PATCH/DELETE 更新或删除评论，提交后无法撤回，因此确认环节不可省略。

4. 若用户选择"选择性提交"，仅提交用户指定的编号对应意见。

### 阶段 3：提交行级 diff_comment

**关键**：GitCode 行级评论必须用 `position` 字段（不是 `line`/`side`/`diff_side`），否则降级为 PR 级普通评论。

读 [references/gitcode-review-api.md](references/gitcode-review-api.md)（**必读**，含字段对照表和踩坑记录），然后：

1. **验证 token**：从 git remote 提取
   ```bash
   TOKEN=$(git remote get-url origin | sed -nE 's|.*oauth2:([^@]+)@.*|\1|p')
   curl -s -o /dev/null -w "%{http_code}" "https://gitcode.com/api/v5/user?access_token=${TOKEN}"
   # 200 = 可用
   ```
2. **确认行号**：行号是 PR head 文件中的绝对行号（`git show $PR_HEAD:<file> | grep -n <pattern>`）。
3. **提交**（二选一）：

   **方式 A：脚本批量提交**（推荐，4 条以上）
   ```bash
   python3 .agents/skills/pypto-machine-code-review/scripts/submit_review_comments.py \
     --pr 4974 --token "$TOKEN" \
     --comment path:position:body \
     --comment path2:pos2:body2
   ```
   见 [scripts/submit_review_comments.py](scripts/submit_review_comments.py)。

   **方式 B：curl 单条提交**
   ```bash
   curl -s -H "PRIVATE-TOKEN: ${TOKEN}" -H "Content-Type: application/json" \
     -X POST "https://gitcode.com/api/v5/repos/cann/pypto/pulls/<PR>/comments" \
     -d "$(jq -n --arg b "$BODY" --arg p "$PATH" --argjson pos "$LINE" \
       '{body:$b, path:$p, position:$pos, need_to_resolve:true}')"
   ```

4. **验证**：确认 `comment_type=diff_comment` 且 `diff_position` 非空
   ```bash
   curl -s -H "PRIVATE-TOKEN: ${TOKEN}" \
     "https://gitcode.com/api/v5/repos/cann/pypto/pulls/<PR>/comments?per_page=50" | \
     python3 -c "import sys,json; [print(c['id'],c.get('comment_type'),c.get('diff_position')) for c in json.load(sys.stdin)]"
   ```

### 阶段 4：报告

输出检视意见汇总表（文件:行 / 级别 / 问题 / comment_id），并给出 PR 链接。

## 硬约束

### GitCode 行级评论

| 必须 | 禁止 |
|---|---|
| **提交前用户确认（阶段 2.5）** | **未确认直接提交**（GitCode 不支持删除/更新） |
| 行级评论用 `position` 字段 | 用 `line`/`side`/`diff_side`（降级为普通评论） |
| 行号取 PR head 绝对行号 | 用 base 行号或 diff 相对行号 |
| 验证 `comment_type=diff_comment` | 假定 201 = 行级评论（可能只是 pr_comment） |
| body 含级别标签（Blocker/Major/Minor） | 无级别裸意见 |
| 聚焦 machine 侧 C++ | 把 docs/tests 改动当检视对象 |
| 先读 checklist 再下结论 | 凭直觉/记忆评审 |
| 意见需有代码证据支撑 | 沿用模式未验证即判定（假阳性） |

### AICPU 热路径性能（§6 命中时）

| 必须 | 禁止 |
|---|---|
| 定长数组 / `Vector`+workspace / `ItemPool` / `SPSC` | 热路径 STL、业务 `new/delete` |
| 显式 `T*`；局部按引用传递 | `uint64_t` 当万能地址；无必要 `&local` |
| trivial 默构 + 合理布局 | 默构清大数组；热点结构塞冷字段 |
| Host 预计算；循环外提 | AICPU 重复 if/算本可 Host 完成的事 |
| 必须 H2D 时用 `NormalizedRtMemcpy` | Host 随意新增直连 `rtMemcpy` |
| `Data()` / `&At(0)` 指针迭代；热循环 `DEV_VERBOSE_DEBUG` | 热循环 `operator[]` / 每轮 `At`；热循环 `DEV_INFO` |

违反 §6 / `perf-rules.md` §2.1–§2.5、§2.9、§2.11、§2.10.1、§2.14：**即使功能正确，也视为回归。**

### 时序 / 重叠流水（§7 命中时）

| 必须 | 禁止 |
|---|---|
| 按读者集合判定：仅 ctrl 串行使用的 DevProg 字段允许写 | 把 DevProg 一律只读，或把 ctrl 私有字段当跨流水竞态 |
| 跨流水读者的本轮状态进 `DevStartArgs` / slot | 按轮次原地改 `nrValidAic` / `scheCpuNum` / sharedBuffer 任务槽 |
| 回收等 AICore；STOP 可见 → 清槽 → barrier → GOODBYE | 只等 sche 就复用 slot；fence 放进可选分支 |
| 硬失败且等其他 AICPU 的等待 ≥ OS 节流窗口 | 硬失败路径继续用远小于节流窗口的短超时 |

违反 §7 / `timing-race-rules.md`：**即使功能正确，也视为常稳回归。**

## 参考文件

| File | Purpose | Load Timing |
|------|---------|-------------|
| [references/review-checklist.md](references/review-checklist.md) | machine 侧 C++ 检视清单（7 大类，含 AICPU 热路径与时序重叠） | 阶段 2 开始前必读 |
| [references/perf-rules.md](references/perf-rules.md) | AICPU 热路径性能规则（原则全文，含 ARMv8 特化） | §6 命中时必读 |
| [references/timing-race-rules.md](references/timing-race-rules.md) | 时序 / 重叠流水 / OS 节流（按读者集合、先清后放、硬失败等待） | §7 命中时必读 |
| [references/memory-rules.md](references/memory-rules.md) | 内存规则（跨核数据一致性 U4–U7 等） | 阶段 2 与内置清单合并执行 |
| [references/gitcode-review-api.md](references/gitcode-review-api.md) | GitCode 行级评论 API 字段对照与踩坑 | 阶段 3 提交前必读 |
| [scripts/submit_review_comments.py](scripts/submit_review_comments.py) | 批量提交 diff_comment 脚本 | 阶段 3 方式 A |
