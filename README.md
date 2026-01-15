# Freqtrade strategies

This Git repo contains free buy/sell strategies for [Freqtrade](https://github.com/freqtrade/freqtrade).

All strategies should work with a freqtrade version of 2022.4 or newer.

## Disclaimer

These strategies are for educational purposes only. Do not risk money 
which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE 
AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING 
RESULTS. 

Always start by testing strategies with a backtesting then run the 
trading bot in Dry-run. Do not engage money before you understand how 
it works and what profit/loss you should expect.

We strongly recommend you to have coding and Python knowledge. Do not 
hesitate to read the source code and understand the mechanism of this 
bot.

## Table of Content

- [Free trading strategies](#free-trading-strategies)
- [Parallel Backtest Tool](#parallel-backtest-tool)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Parameters](#parameters)
    - [Examples](#examples)
- [Contribute](#share-your-own-strategies-and-contribute-to-this-repo)
- [FAQ](#faq)
    - [What is Freqtrade?](#what-is-freqtrade)
    - [What includes these strategies?](#what-includes-these-strategies)
    - [How to install a strategy?](#how-to-install-a-strategy)
    - [How to test a strategy?](#how-to-test-a-strategy)
    - [How to create/optimize a strategy?](https://www.freqtrade.io/en/latest/strategy-customization/)

## Free trading strategies

Value below are result from backtesting from 2018-01-10 to 2018-01-30 and  
`exit_profit_only` enabled. More detail on each strategy page.

|  Strategy | Buy count | AVG profit % | Total profit | AVG duration | Backtest period |
|-----------|-----------|--------------|--------------|--------------|-----------------|
| [Strategy 001](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/Strategy001.py) | 55 | 0.05 | 0.00012102 |  476.1 | 2018-01-10 to 2018-01-30 |
| [Strategy 002](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/Strategy002.py) | 9 | 3.21 | 0.00114807 |  189.4 | 2018-01-10 to 2018-01-30 |
| [Strategy 003](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/Strategy003.py) | 14 | 1.47 | 0.00081740 |  227.5 | 2018-01-10 to 2018-01-30 | 
| [Strategy 004](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/Strategy004.py) | 37 | 0.69 | 0.00102128 |  367.3 | 2018-01-10 to 2018-01-30 | 
| [Strategy 005](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/Strategy005.py) | 180 | 1.16 | 0.00827589 |  156.2 | 2018-01-10 to 2018-01-30 |

Strategies from this repo are free to use. Feel free to update them to your likings.
Most of them  were designed from Hyperopt calculations.

Some only work in specific market conditions, while others are more "general purpose" strategies.
It's noteworthy that depending on the exchange and Pairs used, further optimization can bring better results.

Please keep in mind, results will heavily depend on the pairs, timeframe and timerange used to backtest - so please run your own backtests that mirror your usecase, to evaluate each strategy for yourself.

The results above should serve as a general outline to demonstrate the number of trades to expect. Actual performance will be different based on various factors.

## Parallel Backtest Tool

并行回测工具 - 通过为每个交易对启动独立的回测进程实现并行化，显著减少总回测时间。

### Installation

1. 确保已安装 Python 3.8+ 和 Freqtrade

2. 安装依赖：

```bash
# 创建并激活虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装自定义依赖
pip install -r requirements-custom.txt
```

### Usage

基本用法：

```bash
python -m parallel_backtest --config <config_file> --strategy <strategy_name>
```

### Parameters

| 参数 | 简写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--config` | `-c` | ✓ | - | Freqtrade 配置文件路径 |
| `--strategy` | `-s` | ✓ | - | 策略名称 |
| `--timerange` | - | - | - | 回测时间范围 (如 `20240101-20241231`) |
| `--workers` | `-w` | - | CPU核心数-1 | 最大并发工作进程数 |
| `--output` | `-o` | - | `user_data/backtest_results` | 输出目录 |
| `--pairs` | `-p` | - | 配置文件中的 pair_whitelist | 交易对列表 |
| `--timeout` | - | - | 3600 | 单个回测超时时间（秒） |
| `--debug` | - | - | false | 调试模式，保留临时文件 |

使用 `--` 分隔符可以将额外参数透传给 Freqtrade。

### Examples

```bash
# 基本用法 - 使用配置文件中的交易对
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1

# 指定交易对（空格分隔）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --pairs BTC/USDT ETH/USDT SOL/USDT

# 指定交易对（逗号分隔）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --pairs "BTC/USDT,ETH/USDT,SOL/USDT"

# 指定时间范围和工作进程数
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --timerange 20240101-20241231 --workers 4

# 指定输出路径
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --output user_data/backtest_results/my_result.json

# 调试模式（保留临时文件）
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    --debug

# 透传 Freqtrade 参数
python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1 \
    -- --cache none --enable-protections
```

### Features

- **并行执行**: 为每个交易对启动独立的回测进程，充分利用多核 CPU
- **资源隔离**: 每个工作进程使用独立的配置文件、输出目录和日志文件
- **结果合并**: 自动合并所有回测结果，生成 Freqtrade 兼容的统一报告
- **进度监控**: 实时显示进度条和已完成任务数
- **错误容错**: 单个交易对失败不影响其他任务，最终报告显示失败详情
- **优雅中断**: 支持 Ctrl+C 优雅终止，保存部分结果

## Share your own strategies and contribute to this repo

Feel free to send your strategies, comments, optimizations and pull requests via an 
[Issue ticket](https://github.com/freqtrade/freqtrade-strategies/issues/new) or as a [Pull request](https://github.com/freqtrade/freqtrade-strategies/pulls) enhancing this repository.

## FAQ

### What is Freqtrade?

[Freqtrade](https://github.com/freqtrade/freqtrade) Freqtrade is a free and open source crypto trading bot written in Python.
It is designed to support all major exchanges and be controlled via Telegram. It contains backtesting, plotting and money management tools as well as strategy optimization by machine learning.

### What includes these strategies?

Each Strategies includes:  

- [x] **Minimal ROI**: Minimal ROI optimized for the strategy.
- [x] **Stoploss**: Optimal stoploss.
- [x] **Buy signals**: Result from Hyperopt or based on exisiting trading strategies.
- [x] **Sell signals**: Result from Hyperopt or based on exisiting trading strategies.
- [x] **Indicators**: Includes the indicators required to run the strategy.

Best backtest multiple strategies with the exchange and pairs you're interrested in, and finetune the strategy to the markets you're trading.

### How to install a strategy?

First you need a [working Freqtrade](https://freqtrade.io).

Once you have the bot on the right version, follow this steps:

1. Select the strategy you want. All strategies of the repo are into 
[user_data/strategies](https://github.com/freqtrade/freqtrade-strategies/tree/main/user_data/strategies)
2. Copy the strategy file
3. Paste it into your `user_data/strategies` folder
4. Run the bot with the parameter `--strategy <STRATEGY CLASS NAME>` (ex: `freqtrade trade --strategy Strategy001`)

More information [about backtesting](https://www.freqtrade.io/en/latest/backtesting/) and [strategy customization](https://www.freqtrade.io/en/latest/strategy-customization/).

### How to test a strategy?

Let assume you have selected the strategy `strategy001.py`:

#### Simple backtesting

```bash
freqtrade backtesting --strategy Strategy001
```

#### Refresh your test data

```bash
freqtrade download-data --days 100
```

*Note:* Generally, it's recommended to use static backtest data (from a defined period of time) for comparable results.

Please check out the [official backtesting documentation](https://www.freqtrade.io/en/latest/backtesting/) for more information.
