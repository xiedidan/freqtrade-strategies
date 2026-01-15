# 设计文档

## 概述

本设计文档描述了 Freqtrade 并行回测工具的技术架构和实现方案。该工具使用 Python 的 `concurrent.futures` 模块实现多进程并行，通过 `subprocess` 调用 Freqtrade CLI 执行独立回测，最后合并所有结果为 Freqtrade 兼容的格式。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Backtest Tool                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   CLI       │  │   Config    │  │   Progress Monitor      │  │
│  │   Parser    │──│   Generator │  │   (tqdm)                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                      │                │
│         ▼                ▼                      ▼                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Task Executor                             ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │              ProcessPoolExecutor                        │││
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │││
│  │  │  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │       │││
│  │  │  │BTC/USDT │ │ETH/USDT │ │SOL/USDT │ │ ...     │       │││
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Result Merger                             ││
│  │  - 合并 trades 列表                                          ││
│  │  - 重新计算统计指标                                          ││
│  │  - 生成 Freqtrade 兼容的输出文件                             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 组件和接口

### 1. CLI Parser (命令行解析器)

```python
class CLIParser:
    """Parse command line arguments for parallel backtest tool."""
    
    def parse_args(self, args: List[str]) -> BacktestConfig:
        """
        Parse CLI arguments and return configuration.
        
        Args:
            args: Command line arguments
            
        Returns:
            BacktestConfig with all parsed options
        """
        pass
```

**支持的参数:**
- `--config`: Freqtrade 配置文件路径 (必需)
- `--strategy`: 策略名称 (必需)
- `--timerange`: 回测时间范围 (可选)
- `--workers`: 最大并发数 (默认: CPU核心数 - 1)
- `--output`: 输出目录 (默认: user_data/backtest_results)
- `--pairs`: 交易对列表 (可选，覆盖配置文件)
- `--timeout`: 单个回测超时时间秒数 (默认: 3600)
- `--debug`: 调试模式，保留临时文件
- 其他参数透传给 Freqtrade

### 2. Config Generator (配置生成器)

```python
class ConfigGenerator:
    """Generate isolated config files for each worker."""
    
    def __init__(self, base_config_path: str, temp_dir: str):
        """
        Initialize config generator.
        
        Args:
            base_config_path: Path to base Freqtrade config
            temp_dir: Directory for temporary files
        """
        pass
    
    def generate_worker_config(self, pair: str, worker_id: int) -> WorkerConfig:
        """
        Generate isolated config for a single pair.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            worker_id: Unique worker identifier
            
        Returns:
            WorkerConfig with paths to isolated resources
        """
        pass
```

**WorkerConfig 结构:**
```python
@dataclass
class WorkerConfig:
    pair: str                    # Trading pair
    config_path: str             # Path to temporary config file
    result_dir: str              # Isolated result directory
    log_file: str                # Isolated log file path
    worker_id: int               # Unique worker ID
```

**隔离策略:**
- 临时配置文件: `temp/worker_{id}/config.json`
- 结果目录: `temp/worker_{id}/results/`
- 日志文件: `temp/worker_{id}/freqtrade.log`

### 3. Backtest Worker (回测工作进程)

```python
class BacktestWorker:
    """Execute single pair backtest using Freqtrade CLI."""
    
    def run(self, worker_config: WorkerConfig, 
            strategy: str, 
            timerange: Optional[str],
            extra_args: List[str]) -> WorkerResult:
        """
        Run backtest for a single pair.
        
        Args:
            worker_config: Isolated worker configuration
            strategy: Strategy name
            timerange: Backtest time range
            extra_args: Additional Freqtrade arguments
            
        Returns:
            WorkerResult with backtest outcome
        """
        pass
```

**WorkerResult 结构:**
```python
@dataclass
class WorkerResult:
    pair: str                    # Trading pair
    success: bool                # Whether backtest succeeded
    result_file: Optional[str]   # Path to result JSON (if success)
    error_message: Optional[str] # Error message (if failed)
    duration: float              # Execution time in seconds
    trades_count: int            # Number of trades
    profit_ratio: float          # Total profit ratio
```

### 4. Task Executor (任务执行器)

```python
class TaskExecutor:
    """Manage parallel execution of backtest tasks."""
    
    def __init__(self, max_workers: int, timeout: int):
        """
        Initialize task executor.
        
        Args:
            max_workers: Maximum concurrent workers
            timeout: Timeout per task in seconds
        """
        pass
    
    def execute_all(self, tasks: List[BacktestTask], 
                    progress_callback: Callable) -> List[WorkerResult]:
        """
        Execute all backtest tasks in parallel.
        
        Args:
            tasks: List of backtest tasks
            progress_callback: Callback for progress updates
            
        Returns:
            List of worker results
        """
        pass
```

### 5. Result Merger (结果合并器)

```python
class ResultMerger:
    """Merge multiple backtest results into unified format."""
    
    def merge(self, results: List[WorkerResult], 
              output_path: str,
              strategy_name: str) -> MergedResult:
        """
        Merge all successful results into single file.
        
        Args:
            results: List of worker results
            output_path: Output file path
            strategy_name: Strategy name for result
            
        Returns:
            MergedResult with summary statistics
        """
        pass
```

**合并逻辑:**

1. **交易记录合并:**
   - 收集所有 `trades` 列表
   - 按 `open_timestamp` 排序
   - 保留所有原始字段

2. **统计指标重新计算:**
   - `total_trades`: 所有交易总数
   - `profit_total_abs`: 所有利润绝对值之和
   - `profit_total`: 利润率 = profit_total_abs / starting_balance
   - `wins/losses/draws`: 重新统计
   - `winrate`: wins / total_trades
   - `max_drawdown_*`: 基于合并后的交易序列重新计算

3. **每对统计 (results_per_pair):**
   - 保留各交易对的独立统计

4. **输出格式:**
   - 生成与 Freqtrade 兼容的 JSON 结构
   - 创建 `.meta.json` 文件
   - 可选: 打包为 `.zip` 文件

## 数据模型

### BacktestConfig (回测配置)

```python
@dataclass
class BacktestConfig:
    config_path: str              # Base config file path
    strategy: str                 # Strategy name
    pairs: List[str]              # Trading pairs to backtest
    timerange: Optional[str]      # Time range for backtest
    max_workers: int              # Maximum concurrent workers
    output_dir: str               # Output directory
    timeout: int                  # Timeout per task (seconds)
    debug: bool                   # Debug mode flag
    extra_args: List[str]         # Extra Freqtrade arguments
```

### FreqtradeResult (Freqtrade 结果格式)

```python
# Freqtrade 回测结果 JSON 结构
{
    "strategy": {
        "<strategy_name>": {
            "trades": [...],           # 交易记录列表
            "locks": [...],            # 锁定记录
            "best_pair": {...},        # 最佳交易对
            "worst_pair": {...},       # 最差交易对
            "results_per_pair": {...}, # 每对结果
            "total_trades": int,       # 总交易数
            "profit_total": float,     # 总利润率
            "profit_total_abs": float, # 总利润绝对值
            "winrate": float,          # 胜率
            # ... 其他统计字段
        }
    },
    "strategy_comparison": [...]       # 策略比较
}
```

### Trade (交易记录)

```python
@dataclass
class Trade:
    pair: str                     # Trading pair
    stake_amount: float           # Stake amount
    amount: float                 # Trade amount
    open_date: str                # Open date ISO format
    close_date: str               # Close date ISO format
    open_rate: float              # Entry price
    close_rate: float             # Exit price
    profit_ratio: float           # Profit ratio
    profit_abs: float             # Profit absolute
    exit_reason: str              # Exit reason
    trade_duration: int           # Duration in minutes
    is_short: bool                # Short trade flag
    open_timestamp: int           # Open timestamp (ms)
    close_timestamp: int          # Close timestamp (ms)
    # ... 其他字段
```


## 正确性属性

*正确性属性是指在系统所有有效执行中都应该保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性作为人类可读规格和机器可验证正确性保证之间的桥梁。*

### Property 1: 交易对解析一致性

*For any* 命令行参数中的 `--pairs` 列表，解析后创建的回测任务数量应该等于交易对数量，且每个任务对应一个唯一的交易对。

**Validates: Requirements 1.1, 1.2**

### Property 2: 资源隔离唯一性

*For any* 生成的 N 个 WorkerConfig，所有配置文件路径、输出目录路径、日志文件路径应该两两不同（共 3N 个路径，无重复）。

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 3: 并发数限制

*For any* 配置的 max_workers 值 W 和任务列表，在任意时刻同时运行的工作进程数不应超过 W。

**Validates: Requirements 1.3**

### Property 4: 交易记录完整性

*For any* 成功的回测结果列表，合并后的 trades 列表长度应该等于所有单独结果 trades 列表长度之和。

**Validates: Requirements 3.1, 3.2**

### Property 5: 统计数据一致性

*For any* 合并后的结果，`total_trades` 字段应该等于 `trades` 列表的长度，`wins + losses + draws` 应该等于 `total_trades`。

**Validates: Requirements 3.3**

### Property 6: 部分失败容错

*For any* 包含失败结果的结果列表，合并操作应该成功完成，且合并后的 trades 只包含成功结果的交易。

**Validates: Requirements 3.5, 4.1**

### Property 7: CLI 参数解析正确性

*For any* 有效的命令行参数组合，解析后的 BacktestConfig 应该正确反映所有参数值，未指定的参数应使用默认值。

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**

## 错误处理

### 错误类型和处理策略

| 错误类型 | 处理策略 | 用户反馈 |
|---------|---------|---------|
| 配置文件不存在 | 立即退出 | 显示错误信息和正确用法 |
| 配置文件格式错误 | 立即退出 | 显示 JSON 解析错误位置 |
| 交易对列表为空 | 立即退出 | 提示需要指定交易对 |
| 单个回测失败 | 记录错误，继续其他任务 | 在摘要中显示失败的交易对 |
| 单个回测超时 | 终止进程，标记失败 | 显示超时的交易对 |
| 所有回测失败 | 退出，不生成合并结果 | 显示所有错误信息 |
| Ctrl+C 中断 | 优雅终止所有进程 | 显示已完成的任务和部分结果 |
| 磁盘空间不足 | 记录错误，尝试清理 | 提示磁盘空间问题 |

### 信号处理

```python
def setup_signal_handlers(executor: TaskExecutor):
    """Setup graceful shutdown on SIGINT/SIGTERM."""
    
    def handler(signum, frame):
        print("\nReceived interrupt signal, shutting down...")
        executor.shutdown(wait=False)
        # Save partial results if any
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
```

## 测试策略

### 单元测试

1. **CLI Parser 测试**
   - 测试各种参数组合的解析
   - 测试默认值
   - 测试无效参数的错误处理

2. **Config Generator 测试**
   - 测试配置文件生成
   - 测试路径唯一性
   - 测试配置内容正确性

3. **Result Merger 测试**
   - 测试交易记录合并
   - 测试统计数据计算
   - 测试空结果处理
   - 测试部分失败处理

### 属性测试

使用 `hypothesis` 库进行属性测试，每个属性测试至少运行 100 次迭代。

1. **Property 1 测试**: 生成随机交易对列表，验证解析一致性
2. **Property 2 测试**: 生成随机数量的 worker，验证路径唯一性
3. **Property 4 测试**: 生成随机回测结果，验证合并完整性
4. **Property 5 测试**: 生成随机交易列表，验证统计一致性
5. **Property 6 测试**: 生成包含失败的结果列表，验证容错性
6. **Property 7 测试**: 生成随机参数组合，验证解析正确性

### 集成测试

1. **端到端测试**: 使用真实的 Freqtrade 配置运行小规模并行回测
2. **Freqtrade 兼容性测试**: 验证合并结果可被 `freqtrade backtesting-analysis` 正确读取

## 文件结构

```
parallel_backtest/
├── __init__.py
├── cli.py              # CLI 解析器
├── config.py           # 配置生成器
├── worker.py           # 回测工作进程
├── executor.py         # 任务执行器
├── merger.py           # 结果合并器
├── models.py           # 数据模型
└── utils.py            # 工具函数

tests/
├── __init__.py
├── test_cli.py         # CLI 测试
├── test_config.py      # 配置生成器测试
├── test_merger.py      # 结果合并器测试
├── test_properties.py  # 属性测试
└── conftest.py         # pytest 配置
```

## 使用示例

```bash
# 基本用法 - 使用配置文件中的交易对
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1

# 指定交易对
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --pairs BTC/USDT ETH/USDT SOL/USDT

# 指定时间范围和工作进程数
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --timerange 20240101-20241231 --workers 4

# 指定输出路径
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --output user_data/backtest_results/parallel_result.json

# 调试模式（保留临时文件）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --debug

# 透传 Freqtrade 参数
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    -- --cache none --enable-protections
```
