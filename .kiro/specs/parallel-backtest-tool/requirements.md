# 需求文档

## 简介

本文档定义了 Freqtrade 并行回测工具的需求规格。该工具旨在解决 Freqtrade 原生回测单线程执行慢的问题，通过为每个交易对启动独立的回测进程实现并行化，并最终合并结果以便生成统一的回测报告。

## 术语表

- **Parallel_Backtest_Tool**: 并行回测工具，负责协调多个 Freqtrade 回测进程的执行
- **Backtest_Worker**: 单个回测工作进程，负责执行单个交易对的回测任务
- **Result_Merger**: 结果合并器，负责将多个独立回测结果合并为统一格式
- **Task_Queue**: 任务队列，管理待执行的回测任务
- **Config_Generator**: 配置生成器，为每个工作进程生成独立的配置文件

## 需求列表

### 需求 1: 并行执行

**用户故事:** 作为交易者，我希望能够并行运行多个交易对的回测，以便显著减少总回测时间。

#### 验收标准

1. WHEN 用户通过 `--pairs` 参数提供交易对列表时, THE Parallel_Backtest_Tool SHALL 使用该列表创建独立的回测任务
2. WHEN 用户未提供 `--pairs` 参数时, THE Parallel_Backtest_Tool SHALL 从配置文件的 `pair_whitelist` 中提取交易对列表
3. WHEN 回测任务创建完成后, THE Parallel_Backtest_Tool SHALL 并发执行任务，最大并发数可配置
4. WHEN 未指定工作进程数时, THE Parallel_Backtest_Tool SHALL 默认使用 (CPU核心数 - 1) 个工作进程
5. WHILE 工作进程执行期间, THE Parallel_Backtest_Tool SHALL 显示实时进度信息，包括已完成/总任务数和预计剩余时间

### 需求 2: 资源隔离

**用户故事:** 作为交易者，我希望每个并行回测都有独立的资源，以避免并发进程之间的冲突。

#### 验收标准

1. WHEN 创建回测任务时, THE Config_Generator SHALL 为每个交易对生成唯一的临时配置文件
2. WHEN Backtest_Worker 启动时, THE Parallel_Backtest_Tool SHALL 分配唯一的输出目录存放回测结果
3. WHEN Backtest_Worker 启动时, THE Parallel_Backtest_Tool SHALL 分配唯一的日志文件路径以防止日志冲突
4. WHEN 所有工作进程完成后, THE Parallel_Backtest_Tool SHALL 清理临时配置文件，除非启用了调试模式

### 需求 3: 结果合并

**用户故事:** 作为交易者，我希望将所有并行回测结果合并为单一的统一结果，以便使用 Freqtrade 内置的报告工具。

#### 验收标准

1. WHEN 所有回测任务成功完成后, THE Result_Merger SHALL 将所有独立的结果 JSON 文件合并为单一的合并结果文件
2. WHEN 合并结果时, THE Result_Merger SHALL 保留所有交易记录，包含正确的时间戳和交易对信息
3. WHEN 合并结果时, THE Result_Merger SHALL 基于合并后的交易重新计算汇总统计数据（总利润、胜率、回撤等）
4. THE 合并后的结果文件 SHALL 兼容 Freqtrade 的 `backtesting-analysis` 和 `plot-profit` 命令
5. IF 任何单个回测失败, THEN THE Result_Merger SHALL 仍然合并成功的结果并报告哪些交易对失败

### 需求 4: 错误处理

**用户故事:** 作为交易者，我希望有健壮的错误处理机制，以便单个失败不会导致整个并行回测崩溃。

#### 验收标准

1. IF Backtest_Worker 遇到错误, THEN THE Parallel_Backtest_Tool SHALL 记录错误并继续执行剩余任务
2. IF Backtest_Worker 超时, THEN THE Parallel_Backtest_Tool SHALL 终止该进程并将任务标记为失败
3. WHEN 用户按下 Ctrl+C 时, THE Parallel_Backtest_Tool SHALL 优雅地终止所有运行中的工作进程并保存部分结果
4. WHEN 所有任务完成后, THE Parallel_Backtest_Tool SHALL 显示摘要，包括成功、失败和跳过的任务数量

### 需求 5: 命令行接口

**用户故事:** 作为交易者，我希望有一个类似 Freqtrade 的简单 CLI 接口，以便轻松将此工具集成到我的工作流程中。

#### 验收标准

1. THE Parallel_Backtest_Tool SHALL 接受 `--config` 参数指定基础 Freqtrade 配置文件
2. THE Parallel_Backtest_Tool SHALL 接受 `--strategy` 参数指定策略名称
3. THE Parallel_Backtest_Tool SHALL 接受 `--timerange` 参数指定回测时间范围
4. THE Parallel_Backtest_Tool SHALL 接受 `--workers` 参数指定最大并发工作进程数
5. THE Parallel_Backtest_Tool SHALL 接受 `--output` 参数指定合并结果的输出路径
6. THE Parallel_Backtest_Tool SHALL 接受 `--pairs` 参数指定交易对列表（格式与 Freqtrade 一致，如 `--pairs BTC/USDT ETH/USDT` 或 `--pairs "BTC/USDT,ETH/USDT"`）
7. WHERE 提供了 `--pairs` 参数时, THE Parallel_Backtest_Tool SHALL 优先使用命令行指定的交易对，忽略配置文件中的 `pair_whitelist`
8. THE Parallel_Backtest_Tool SHALL 将额外的 Freqtrade 回测参数透传给每个工作进程

### 需求 6: 进度监控

**用户故事:** 作为交易者，我希望能够监控并行回测的进度，以便了解整个过程需要多长时间。

#### 验收标准

1. WHILE 回测运行期间, THE Parallel_Backtest_Tool SHALL 显示进度条展示整体完成百分比
2. WHILE 回测运行期间, THE Parallel_Backtest_Tool SHALL 显示当前正在处理的交易对
3. WHEN 任务完成时, THE Parallel_Backtest_Tool SHALL 显示该交易对的结果摘要（利润、交易次数）
4. WHEN 所有任务完成后, THE Parallel_Backtest_Tool SHALL 显示总执行时间和相比顺序执行的加速比
