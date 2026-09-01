# AGENTS.md

## 项目目标

这是一个已经存在并持续维护的软件项目。

你的主要任务是在现有代码基础上开发新的 feature，
而不是重新设计或重写现有系统。

## 开发规则

修改代码之前：

1. 先阅读并理解相关的现有实现。
2. 在 repository 中搜索是否已经存在类似实现，优先复用已有代码。
3. 如果修改涉及多个 module，先理清相关 call path 和 data flow。

修改代码时：

1. 遵循最小改动原则，只修改完成当前任务所必要的代码。
2. 不要主动 refactor 与当前任务无关的代码。
3. 不要无必要地创建新的 wrapper、fallback、V2 implementation。
4. 除非任务明确要求，否则不要改变现有 interface 和 behavior。
5. 遵循项目已有的 architecture、naming、logging 和 configuration 方式。

## 不确定时

不要猜测项目行为。

优先阅读：
- implementation
- call sites
- configuration
- tests

如果仍然存在可能影响 architecture 的重大歧义，再询问用户。