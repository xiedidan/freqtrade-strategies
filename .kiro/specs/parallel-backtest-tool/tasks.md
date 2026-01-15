# 实现计划: 并行回测工具

## 概述

本实现计划将并行回测工具的设计分解为可执行的编码任务。使用 Python 实现，依赖 `concurrent.futures` 进行并行处理，`tqdm` 显示进度，`hypothesis` 进行属性测试。

## 任务列表

- [x] 1. 项目结构和数据模型
  - [x] 1.1 创建项目目录结构和 `__init__.py` 文件
    - 创建 `parallel_backtest/` 目录
    - 创建所有模块文件的空白版本
    - _Requirements: 项目基础结构_

  - [x] 1.2 实现数据模型 (`models.py`)
    - 实现 `BacktestConfig` 数据类
    - 实现 `WorkerConfig` 数据类
    - 实现 `WorkerResult` 数据类
    - 实现 `MergedResult` 数据类
    - _Requirements: 数据模型定义_

- [x] 2. CLI 解析器
  - [x] 2.1 实现命令行参数解析 (`cli.py`)
    - 使用 `argparse` 实现参数解析
    - 支持 `--config`, `--strategy`, `--timerange`, `--workers`, `--output`, `--pairs`, `--timeout`, `--debug` 参数
    - 实现参数验证和默认值
    - 实现额外参数透传 (使用 `--` 分隔)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 2.2 编写 CLI 解析器属性测试
    - **Property 7: CLI 参数解析正确性**
    - **Validates: Requirements 5.1-5.8**

- [x] 3. 配置生成器
  - [x] 3.1 实现配置生成器 (`config.py`)
    - 实现 `ConfigGenerator` 类
    - 实现从基础配置读取交易对列表
    - 实现为每个交易对生成独立配置文件
    - 实现临时目录管理
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3_

  - [x] 3.2 编写配置生成器属性测试
    - **Property 1: 交易对解析一致性**
    - **Property 2: 资源隔离唯一性**
    - **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3**

- [x] 4. 检查点 - 确保所有测试通过
  - 运行现有测试，确保基础组件正常工作
  - 如有问题请询问用户

- [x] 5. 回测工作进程
  - [x] 5.1 实现回测工作进程 (`worker.py`)
    - 实现 `BacktestWorker` 类
    - 使用 `subprocess` 调用 Freqtrade CLI
    - 实现结果文件解析
    - 实现错误捕获和超时处理
    - _Requirements: 4.1, 4.2_

  - [x] 5.2 编写工作进程单元测试
    - 测试命令构建
    - 测试结果解析
    - 测试错误处理
    - _Requirements: 4.1, 4.2_

- [x] 6. 任务执行器
  - [x] 6.1 实现任务执行器 (`executor.py`)
    - 实现 `TaskExecutor` 类
    - 使用 `ProcessPoolExecutor` 实现并行执行
    - 实现进度回调
    - 实现优雅关闭 (信号处理)
    - _Requirements: 1.3, 1.4, 4.3_

  - [x] 6.2 编写任务执行器属性测试
    - **Property 3: 并发数限制**
    - **Validates: Requirements 1.3**

- [x] 7. 检查点 - 确保所有测试通过
  - 运行现有测试，确保执行器正常工作
  - 如有问题请询问用户

- [x] 8. 结果合并器
  - [x] 8.1 实现结果合并器 (`merger.py`)
    - 实现 `ResultMerger` 类
    - 实现交易记录合并 (按时间戳排序)
    - 实现统计数据重新计算
    - 实现 Freqtrade 兼容的 JSON 输出
    - 实现 `.meta.json` 文件生成
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 8.2 编写结果合并器属性测试
    - **Property 4: 交易记录完整性**
    - **Property 5: 统计数据一致性**
    - **Property 6: 部分失败容错**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

- [x] 9. 主程序和进度显示
  - [x] 9.1 实现主程序入口 (`__main__.py`)
    - 整合所有组件
    - 实现 `tqdm` 进度条显示
    - 实现任务完成时的结果摘要输出
    - 实现最终统计信息输出 (执行时间、加速比)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 9.2 实现工具函数 (`utils.py`)
    - 实现临时文件清理
    - 实现日志配置
    - 实现时间格式化
    - _Requirements: 2.4_

- [x] 10. 检查点 - 端到端测试
  - 使用真实配置运行小规模并行回测
  - 验证合并结果可被 Freqtrade 读取
  - 如有问题请询问用户

- [x] 11. 文档和收尾
  - [x] 11.1 更新 README 添加并行回测工具使用说明
    - 添加安装依赖说明
    - 添加使用示例
    - 添加参数说明
    - _Requirements: 文档_

## 注意事项

- 所有任务都是必需的，包括属性测试
- 每个任务都引用了具体的需求以便追溯
- 检查点用于确保增量验证
- 属性测试验证通用正确性属性
- 单元测试验证具体示例和边界情况
