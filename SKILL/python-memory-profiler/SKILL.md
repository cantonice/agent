---

name: python-memory-profiler
description: 分析 Python 程序运行过程中的内存使用情况。当需要排查内存持续增长、峰值内存异常、OOM、疑似内存泄漏、内存没有释放，或者希望记录程序运行全过程的内存变化时使用。该 Skill 不预设具体框架或内存来源，而是从进程级观察开始，根据实际证据逐步定位内存占用来源。
---

# Python 内存分析

分析 Python 程序运行期间的内存变化，并逐步定位内存占用来源。

核心目标不是只得到：

> 程序最大使用了多少内存？

而是进一步回答：

* 内存随程序运行如何变化？
* 内存从什么时候开始增长？
* Peak Memory 出现在什么阶段？
* 哪个进程占用了内存？
* 哪段代码或 Allocation 导致了内存增长？
* 内存之后是否释放？
* 是否存在异常的持续增长或 Memory Leak？

## 核心原则

### 1. 先观察，再定位

不要预先假设内存来自某个 Python Library、Framework 或数据结构。

首先从操作系统和进程层面观察真实内存使用情况，再根据证据决定下一步分析工具。

---

### 2. 从粗到细

默认按照以下层级调查：

```text
程序整体
   ↓
进程
   ↓
时间阶段
   ↓
代码位置
   ↓
Allocation
```

不要一开始就进行逐行 Profiling。

---

### 3. 优先使用外部观察

在能够通过外部工具完成分析时，优先避免修改业务代码。

只有当外部工具无法进一步定位问题时，才考虑：

* 添加 Memory Logging
* 添加 Profiling Code
* 添加 Snapshot
* 修改程序执行逻辑

---

### 4. 区分“现象”和“原因”

例如：

```text
RSS 达到 15 GB
```

这是观测现象。

而：

```text
存在 Memory Leak
```

是原因判断。

不能仅根据 RSS 较高或者 RSS 没有下降，就直接判断存在 Memory Leak。

---

# 主要技术栈

## 1. `/usr/bin/time -v`

用于快速建立程序运行的整体基线。

例如：

```bash
/usr/bin/time -v python script.py
```

重点观察：

* Maximum resident set size
* Elapsed time
* Page faults
* File system input/output

适合回答：

> 这个程序运行一次最大占用了多少内存？

但它只能提供程序运行后的汇总结果，无法描述内存随时间的变化。

因此它通常作为第一层 Baseline，而不是完整的内存分析工具。

---

## 2. `psutil`

用于持续采样 Python 进程运行期间的内存。

重点记录：

```text
timestamp
PID
RSS
VMS
```

必要时同时记录：

```text
Child Process
Total Process Tree RSS
```

适合回答：

> 程序运行过程中，内存到底是怎么变化的？

理想情况下输出类似：

```text
Time        RSS
0s          500 MB
5s          1.2 GB
10s         3.8 GB
15s         8.1 GB
20s         12.4 GB
25s         14.8 GB   ← Peak
30s         9.2 GB
```

通过时间序列首先找到：

```text
什么时候开始增长？
什么时候增长最快？
什么时候达到 Peak？
什么时候下降？
是否持续增长？
```

---

## 3. Linux `/proc`

当需要从操作系统层面进一步观察进程内存时使用。

主要包括：

```text
/proc/<pid>/status
/proc/<pid>/smaps_rollup
/proc/<pid>/smaps
```

其中：

`status`

用于快速查看进程基本内存状态。

`smaps_rollup`

用于查看整个进程 Memory Mapping 的汇总信息。

`smaps`

用于进一步分析具体 Memory Mapping。

不要默认读取完整 `smaps`。

只有当 RSS 时间序列已经表明存在值得进一步调查的内存行为时，再深入分析。

---

## 4. Scalene

当已经确定：

> 某个程序阶段发生明显内存增长

但还不知道：

> 哪段 Python 代码与内存增长有关

时，可以使用 Scalene。

Scalene 主要用于进一步建立：

```text
Memory Growth
      ↓
Code Location
```

之间的关系。

它属于第二阶段 Profiling 工具，而不是默认第一步。

---

## 5. Memray

当需要进一步回答：

> 内存到底在哪里被 Allocation？

或者：

> 哪条调用链产生了大量内存分配？

时使用 Memray。

重点用于：

* Allocation Tracking
* Allocation Call Stack
* Large Allocation
* Repeated Allocation
* Native Allocation
* Memory Leak Investigation

典型流程：

```bash
memray run -o memray.bin script.py
```

然后分析生成的 Profiling 数据。

Memray 属于深入定位工具。

一般在 Process-level Monitoring 已经发现异常后使用。

---

## 6. `tracemalloc`

用于分析 Python Runtime 管理的内存 Allocation。

适合：

* Python Allocation
* Snapshot
* Allocation Difference
* Python Object Growth

但它观察的是特定范围的 Python 内存分配。

因此不要把：

```text
tracemalloc memory
```

直接等价为：

```text
process RSS
```

如果 `tracemalloc` 无法解释实际 RSS 增长，应继续从系统和 Native Allocation 层面调查。

---

# 默认调查流程

遇到 Python 内存问题时，优先执行：

```text
Step 1
/usr/bin/time -v
        ↓
建立整体 Baseline

Step 2
psutil / /proc
        ↓
记录 RSS 时间序列

Step 3
观察 Process Tree
        ↓
确定内存来自哪个进程

Step 4
结合程序日志 / 执行阶段
        ↓
确定内存增长发生在哪个阶段

Step 5
Scalene
        ↓
定位相关代码区域

Step 6
Memray / tracemalloc
        ↓
进一步分析 Allocation
```

不要求每次都执行全部步骤。

如果前面的工具已经能够回答问题，就停止继续增加 Profiling 复杂度。

---

# 推荐的分析思路

看到：

```text
Peak RSS = 15 GB
```

不要直接开始猜测原因。

应该继续调查：

```text
15 GB
 │
 ├─ 是逐渐增长到 15 GB？
 │
 ├─ 还是突然从 5 GB 跳到 15 GB？
 │
 ├─ Peak 持续多久？
 │
 ├─ Peak 之后是否下降？
 │
 ├─ 主进程还是子进程？
 │
 └─ 对应程序正在执行什么阶段？
```

首先建立：

```text
Memory
   ↑
   │               ┌──────
   │          ┌────┘
   │      ┌───┘
   │  ┌───┘
   └────────────────────→ Time
```

然后再把 Memory Timeline 与程序执行阶段对应起来：

```text
Time
 │
 ├── initialization
 ├── loading
 ├── processing
 ├── inference
 ├── post-processing
 └── cleanup
```

最终逐步缩小调查范围。

---

# 输出要求

分析结果应该优先区分“观测结果”和“原因判断”。

推荐：

```text
观测结果

Baseline RSS:
Peak RSS:
Peak Time:
Peak 所处程序阶段:
是否存在 Child Process:
Peak 后是否下降:

内存变化趋势

...

目前能够确定

...

目前不能确定

...

下一步最小调查动作

...
```

不要因为观察到以下现象：

```text
RSS 没有下降
```

就直接得出：

```text
Memory Leak
```

需要进一步证据支持。

---

# Skill 范围

本 Skill 主要用于：

> Linux 环境下 Python 程序运行过程中的内存分析与定位。

核心技术栈：

```text
/usr/bin/time
       ↓
psutil
       ↓
Linux /proc
       ↓
Scalene
       ↓
Memray
       ↓
tracemalloc
```

核心思想：

> 先测量真实的内存行为，再讨论内存来自哪里。

以及：

> 使用能够回答当前问题的最简单工具，不进行没有证据支持的过度 Profiling。
