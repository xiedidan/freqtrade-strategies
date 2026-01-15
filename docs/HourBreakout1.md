# HourBreakout1 策略使用指南

## 策略简介

HourBreakout1 是一个基于 FreqTrade 框架的多时间框架突破割头皮策略。该策略通过分析 1 小时、5 分钟和 1 分钟三个时间框架的价格行为，识别突破-回踩-反弹的交易模式。

### 核心逻辑

1. **突破识别**: 5 分钟收盘价突破前 1 小时最高价
2. **回踩确认**: 1 分钟价格回踩至 MA5 支撑位
3. **反弹入场**: 1 分钟价格从 MA5 反弹时做多入场
4. **风险控制**: 基于 1 小时最高价的止损和基于时间的止盈

---

## 环境配置

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. 安装 FreqTrade

```bash
# 安装 FreqTrade
pip install freqtrade

# 或者从源码安装
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
pip install -e .
```

### 3. 安装依赖

```bash
pip install numpy pandas ta-lib
```

### 4. 验证安装

```bash
freqtrade --version
```

---

## 回测运行

### 1. 下载历史数据

#### 下载一年的数据（推荐）

```powershell
# 下载一年的历史数据（需要 1m, 5m, 1h 三个时间框架）
freqtrade download-data --config configs/HourBreakout1.json --timeframes 1m 5m 1h --days 365

# 指定交易对下载
freqtrade download-data --config configs/HourBreakout1.json --pairs BTC/USDT ETH/USDT SOL/USDT DOGE/USDT BNB/USDT --timeframes 1m 5m 1h --days 365

# 下载更长时间的数据（800天，约2年多）
freqtrade download-data --config configs/HourBreakout1.json --timeframes 1m 5m 1h --days 800
```

**注意事项**：
- 1分钟数据量较大，下载时间可能较长（几分钟到十几分钟）
- 确保有足够的磁盘空间（每个交易对约 500MB-1GB）
- 建议使用稳定的网络连接

### 2. 运行回测

#### 单交易对回测

```powershell
# 基础回测
freqtrade backtesting --config configs/HourBreakout1.json --strategy HourBreakout1 --timerange 20241201-20241231

# 详细回测（包含交易明细）
freqtrade backtesting --config configs/HourBreakout1.json --strategy HourBreakout1 --timerange 20241201-20241231 --export trades

# 快速测试（一周数据）
freqtrade backtesting --config configs/HourBreakout1.json --strategy HourBreakout1 --pairs BTC/USDT --export signals --breakdown month --timerange 20251201-20251207 --cache none
```

#### 多交易对并行回测（推荐）

使用并行回测工具可以显著减少回测时间，特别是在测试多个交易对时：

```powershell
# 安装并行回测工具依赖
pip install -r requirements-custom.txt

# 并行回测多个交易对（使用配置文件中的交易对列表）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 --timerange 20240101-20241231

# 指定交易对并行回测
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 --pairs BTC/USDT ETH/USDT SOL/USDT DOGE/USDT BNB/USDT --timerange 20240101-20241231

# 指定工作进程数（默认为 CPU核心数-1）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 --pairs BTC/USDT ETH/USDT SOL/USDT --timerange 20240101-20241231 --workers 4

# 快速测试（短时间范围）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 --pairs BTC/USDT ETH/USDT --timerange 20241201-20241210 --timeout 600

# 调试模式（保留临时文件）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 --pairs BTC/USDT --timerange 20241201-20241210 --debug
```

**并行回测优势**：
- ⚡ **速度提升**：多交易对回测时间接近单个交易对的时间（加速比约等于交易对数量）
- 🔄 **自动合并**：自动合并所有交易对的结果，生成统一报告
- 📊 **进度监控**：实时显示每个交易对的回测进度和结果
- 🛡️ **容错机制**：单个交易对失败不影响其他交易对

**参数说明**：
- `--config`: 配置文件路径（必需）
- `--strategy`: 策略名称（必需）
- `--pairs`: 交易对列表（可选，不指定则使用配置文件中的列表）
- `--timerange`: 回测时间范围（可选，格式：YYYYMMDD-YYYYMMDD）
- `--workers`: 并发工作进程数（可选，默认：CPU核心数-1）
- `--timeout`: 单个回测超时时间秒数（可选，默认：3600）
- `--output`: 输出目录（可选，默认：user_data/backtest_results）
- `--debug`: 调试模式，保留临时文件（可选）

#### 传统多交易对回测

```powershell
# 多交易对详细回测（按月分解收益）
freqtrade backtesting --config configs/HourBreakout1.json --strategy HourBreakout1 --pairs BTC/USDT ETH/USDT SOL/USDT DOGE/USDT BNB/USDT --export signals --breakdown month --timerange 20240101-20241231 --cache none
```

**注意**：传统方式会顺序执行每个交易对，总时间 = 单个交易对时间 × 交易对数量

### 3. 查看回测结果

```powershell
# 生成回测报告
freqtrade backtesting-analysis --config configs/HourBreakout1.json --analysis-groups 0 --enter-reason-list all --exit-reason-list all
```

--analysis-groups 参数：
0: 入场原因
1: 出场原因
2: 入场+出场组合
3: 交易对
4: 交易对+入场+出场

---

## 超参数优化 (HyperOpt)

### 可优化参数

| 参数名 | 范围 | 默认值 | 说明 |
|--------|------|--------|------|
| ma_period | 3-10 | 5 | MA 周期 |
| exit_minutes | 5-60 | 15 | 时间止盈分钟数 |
| min_breakout_pct | 0.001-0.01 | 0.002 | 最小突破百分比 |
| pullback_tolerance | 0.0001-0.002 | 0.0005 | 回踩容忍度 |
| max_position_hours | 1.0-8.0 | 4.0 | 最大持仓时间（小时） |
| min_volume_threshold | 0.5-3.0 | 1.0 | 最小成交量阈值 |
| stop_loss_buffer_pct | 0.001-0.01 | 0.005 | 止损缓冲百分比 |
| min_entry_spacing | 10-30 | 15 | 最小入场间隔（K线数） |
| breakout_strength_threshold | 0.001-0.005 | 0.002 | 突破强度阈值 |
| rebound_strength_threshold | 0.001-0.01 | 0.003 | 反弹强度阈值 |

### 运行超参数优化

```powershell
# 优化买入参数
freqtrade hyperopt --config configs/HourBreakout1.json --strategy HourBreakout1 --hyperopt-loss SharpeHyperOptLoss --spaces buy --epochs 100 --timerange 20241101-20241231

# 优化卖出参数
freqtrade hyperopt --config configs/HourBreakout1.json --strategy HourBreakout1 --hyperopt-loss SharpeHyperOptLoss --spaces sell --epochs 100

# 同时优化买卖参数
freqtrade hyperopt --config configs/HourBreakout1.json --strategy HourBreakout1 --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --epochs 200

# 使用不同的损失函数
freqtrade hyperopt --config configs/HourBreakout1.json --strategy HourBreakout1 --hyperopt-loss MaxDrawDownHyperOptLoss --spaces buy sell --epochs 100
```

### 应用优化结果

```powershell
# 查看优化结果
freqtrade hyperopt-show --config configs/HourBreakout1.json --best

# 导出最佳参数
freqtrade hyperopt-show --config configs/HourBreakout1.json --best --print-json
```

将优化后的参数更新到 `configs/HourBreakout1.json` 的 `hourbreakout1_params` 部分。

---

## 实盘运行

### 1. 配置交易所 API

编辑 `configs/HourBreakout1.json`，填入交易所 API 密钥：

```json
{
    "exchange": {
        "name": "binance",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET"
    }
}
```

### 2. 模拟交易 (Dry Run)

```powershell
# 启动模拟交易
freqtrade trade --config configs/HourBreakout1.json --strategy HourBreakout1
```

### 3. 实盘交易

```powershell
# 修改配置文件，设置 dry_run 为 false
# "dry_run": false

# 启动实盘交易
freqtrade trade --config configs/HourBreakout1.json --strategy HourBreakout1
```

### 4. 后台运行

```powershell
# Windows 使用 Start-Process
Start-Process -NoNewWindow freqtrade -ArgumentList "trade --config configs/HourBreakout1.json --strategy HourBreakout1"

# 或者使用 Windows 服务/任务计划程序
```

---

## 配置说明

### 主要配置项

```json
{
    "max_open_trades": 3,           // 最大同时持仓数
    "stake_currency": "USDT",       // 计价货币
    "stake_amount": "unlimited",    // 每笔交易金额
    "dry_run": true,                // 模拟交易模式
    "dry_run_wallet": 1000,         // 模拟钱包金额
    
    "minimal_roi": {
        "60": 0.01,                 // 60分钟后 1% 止盈
        "30": 0.02,                 // 30分钟后 2% 止盈
        "15": 0.03,                 // 15分钟后 3% 止盈
        "0": 0.05                   // 立即 5% 止盈
    },
    
    "stoploss": -0.05               // 5% 止损
}
```

### 交易对配置

```json
{
    "exchange": {
        "pair_whitelist": [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT"
        ]
    }
}
```

---

## 监控与日志

### 查看日志

```bash
# 实时查看日志
tail -f freqtrade.log

# 查看最近的交易
freqtrade show-trades --config configs/HourBreakout1.json
```

### 启用 Telegram 通知

编辑配置文件：

```json
{
    "telegram": {
        "enabled": true,
        "token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    }
}
```

---

## 常见问题

### Q: 回测时提示数据不足？

确保下载了所有需要的时间框架数据：

```bash
freqtrade download-data --config configs/HourBreakout1.json --timeframes 1m 5m 1h --days 365
```

### Q: 如何查看策略的详细信号？

```bash
freqtrade backtesting --config configs/HourBreakout1.json \
    --strategy HourBreakout1 \
    --export signals
```

### Q: 超参数优化很慢怎么办？

- 减少 epochs 数量
- 缩短 timerange
- 使用更少的交易对
- 考虑使用 `--jobs -1` 启用多核并行

### Q: 并行回测工具报错怎么办？

1. **找不到 freqtrade 命令**：
   - 确保已激活虚拟环境
   - 工具会自动使用 `python -m freqtrade` 方式调用

2. **所有回测失败**：
   - 使用 `--debug` 参数保留临时文件
   - 检查临时目录中的日志文件（路径会在输出中显示）
   - 确认数据已下载且时间范围正确

3. **回测超时**：
   - 增加 `--timeout` 参数值（默认 3600 秒）
   - 缩短时间范围进行测试
   - 示例：`--timeout 1800` （30分钟）

4. **内存不足**：
   - 减少 `--workers` 数量
   - 缩短时间范围
   - 减少同时回测的交易对数量

### Q: 如何对比并行回测和传统回测的速度？

并行回测工具会在结果中显示加速比：

```
Execution time:  1m 58s
Sequential est:  3m 54s
Speedup:         1.98x
```

一般来说：
- 2个交易对：约 2x 加速
- 4个交易对：约 4x 加速
- 8个交易对：约 8x 加速（取决于 CPU 核心数）

---

## 风险提示

⚠️ **重要声明**：

1. 本策略仅供学习和研究使用
2. 历史回测结果不代表未来收益
3. 请务必先在模拟环境充分测试
4. 不要投入无法承受损失的资金
5. 加密货币交易具有高风险性
