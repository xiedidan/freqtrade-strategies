"""
Result merger for combining multiple backtest results.

This module implements the ResultMerger class that merges individual
backtest results into a unified Freqtrade-compatible format.
"""

import json
import os
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import MergedResult, WorkerResult


class ResultMerger:
    """
    Merge multiple backtest results into unified Freqtrade-compatible format.
    
    This class handles:
    - Loading individual result files (JSON or ZIP)
    - Merging trade records sorted by timestamp
    - Recalculating aggregate statistics
    - Generating Freqtrade-compatible output files
    - Creating .meta.json metadata files
    """
    
    def __init__(self, starting_balance: float = 1000.0, stake_currency: str = "USDT"):
        """
        Initialize the result merger.
        
        Args:
            starting_balance: Initial balance for profit calculations
            stake_currency: Currency used for staking (e.g., "USDT")
        """
        self.starting_balance = starting_balance
        self.stake_currency = stake_currency
    
    def merge(
        self,
        results: List[WorkerResult],
        output_path: str,
        strategy_name: str,
        timerange: Optional[str] = None,
        timeframe: str = "1m"
    ) -> MergedResult:
        """
        Merge all successful results into a single file.
        
        Args:
            results: List of worker results from parallel execution
            output_path: Output file path for merged result
            strategy_name: Strategy name for the result
            timerange: Optional time range string
            timeframe: Timeframe used for backtest
            
        Returns:
            MergedResult with summary statistics
        """
        # Separate successful and failed results
        successful_results = [r for r in results if r.success and r.result_file]
        failed_results = [r for r in results if not r.success]
        
        # Load and merge trade data from successful results
        all_trades: List[Dict[str, Any]] = []
        results_per_pair: List[Dict[str, Any]] = []
        pairlist: List[str] = []
        
        # Track metadata from first successful result
        base_metadata: Dict[str, Any] = {}
        
        for worker_result in successful_results:
            if worker_result.result_file:
                pair_data = self._load_result_file(worker_result.result_file, strategy_name)
                if pair_data:
                    trades = pair_data.get("trades", [])
                    all_trades.extend(trades)
                    pairlist.append(worker_result.pair)
                    
                    # Extract per-pair results
                    pair_results = pair_data.get("results_per_pair", [])
                    results_per_pair.extend(pair_results)
                    
                    # Capture base metadata from first result
                    if not base_metadata:
                        base_metadata = self._extract_base_metadata(pair_data)
        
        # Sort trades by open_timestamp
        all_trades.sort(key=lambda t: t.get("open_timestamp", 0))
        
        # Calculate merged statistics
        stats = self._calculate_statistics(all_trades, pairlist)
        
        # Build merged result structure
        merged_data = self._build_merged_result(
            trades=all_trades,
            stats=stats,
            results_per_pair=results_per_pair,
            pairlist=pairlist,
            strategy_name=strategy_name,
            timerange=timerange,
            timeframe=timeframe,
            base_metadata=base_metadata
        )
        
        # Write output files
        self._write_output_files(merged_data, output_path, strategy_name)
        
        # Build and return MergedResult
        return MergedResult(
            output_file=output_path,
            total_pairs=len(results),
            successful_pairs=len(successful_results),
            failed_pairs=len(failed_results),
            total_trades=stats["total_trades"],
            total_profit_abs=stats["profit_total_abs"],
            total_profit_ratio=stats["profit_total"],
            winrate=stats["winrate"],
            max_drawdown=stats["max_drawdown_abs"],
            failed_pair_names=[r.pair for r in failed_results],
            results_per_pair={
                item["key"]: item for item in results_per_pair
            }
        )
    
    def _load_result_file(
        self, 
        file_path: str, 
        strategy_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load result data from JSON or ZIP file.
        
        Args:
            file_path: Path to result file
            strategy_name: Strategy name to extract
            
        Returns:
            Strategy result data or None if loading fails
        """
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zf:
                    # Get the first JSON file in the archive
                    json_files = [n for n in zf.namelist() if n.endswith(".json")]
                    if json_files:
                        content = zf.read(json_files[0])
                        data = json.loads(content)
                    else:
                        return None
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
            # Extract strategy-specific data
            if "strategy" in data and strategy_name in data["strategy"]:
                return data["strategy"][strategy_name]
            
            return None
        except (json.JSONDecodeError, zipfile.BadZipFile, FileNotFoundError, KeyError):
            return None
    
    def _extract_base_metadata(self, pair_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract base metadata from a result that applies to all pairs.
        
        Args:
            pair_data: Strategy result data from one pair
            
        Returns:
            Dictionary of base metadata fields
        """
        metadata_fields = [
            "stake_amount", "stake_currency", "stake_currency_decimals",
            "starting_balance", "dry_run_wallet", "max_open_trades",
            "max_open_trades_setting", "timeframe", "timeframe_detail",
            "enable_protections", "stoploss", "trailing_stop",
            "trailing_stop_positive", "trailing_stop_positive_offset",
            "trailing_only_offset_is_reached", "use_custom_stoploss",
            "minimal_roi", "use_exit_signal", "exit_profit_only",
            "exit_profit_offset", "ignore_roi_if_entry_signal",
            "trading_mode", "margin_mode"
        ]
        
        return {k: pair_data.get(k) for k in metadata_fields if k in pair_data}
    
    def _calculate_statistics(
        self, 
        trades: List[Dict[str, Any]], 
        pairlist: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from merged trades.
        
        Args:
            trades: List of all merged trades
            pairlist: List of trading pairs
            
        Returns:
            Dictionary of calculated statistics
        """
        total_trades = len(trades)
        
        if total_trades == 0:
            return self._empty_statistics()
        
        # Count wins, losses, draws
        wins = sum(1 for t in trades if t.get("profit_ratio", 0) > 0)
        losses = sum(1 for t in trades if t.get("profit_ratio", 0) < 0)
        draws = total_trades - wins - losses
        
        # Calculate profit metrics
        profit_total_abs = sum(t.get("profit_abs", 0) for t in trades)
        profit_total = profit_total_abs / self.starting_balance if self.starting_balance > 0 else 0
        
        # Calculate winrate
        winrate = wins / total_trades if total_trades > 0 else 0
        
        # Calculate profit mean and median
        profit_ratios = [t.get("profit_ratio", 0) for t in trades]
        profit_mean = sum(profit_ratios) / len(profit_ratios) if profit_ratios else 0
        sorted_ratios = sorted(profit_ratios)
        if len(sorted_ratios) % 2 == 0 and len(sorted_ratios) > 0:
            mid = len(sorted_ratios) // 2
            profit_median = (sorted_ratios[mid - 1] + sorted_ratios[mid]) / 2
        elif len(sorted_ratios) > 0:
            profit_median = sorted_ratios[len(sorted_ratios) // 2]
        else:
            profit_median = 0
        
        # Calculate drawdown
        max_drawdown_abs, max_drawdown_account = self._calculate_drawdown(trades)
        
        # Calculate holding times
        holding_stats = self._calculate_holding_stats(trades)
        
        # Calculate final balance
        final_balance = self.starting_balance + profit_total_abs
        
        # Get backtest time range from trades
        backtest_start_ts = min(t.get("open_timestamp", 0) for t in trades) if trades else 0
        backtest_end_ts = max(t.get("close_timestamp", 0) for t in trades) if trades else 0
        
        # Calculate backtest days
        backtest_days = (backtest_end_ts - backtest_start_ts) / (1000 * 60 * 60 * 24) if backtest_end_ts > backtest_start_ts else 0
        trades_per_day = total_trades / backtest_days if backtest_days > 0 else 0
        
        return {
            "total_trades": total_trades,
            "trade_count_long": sum(1 for t in trades if not t.get("is_short", False)),
            "trade_count_short": sum(1 for t in trades if t.get("is_short", False)),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": winrate,
            "profit_total": profit_total,
            "profit_total_abs": profit_total_abs,
            "profit_total_long": sum(t.get("profit_ratio", 0) for t in trades if not t.get("is_short", False)),
            "profit_total_short": sum(t.get("profit_ratio", 0) for t in trades if t.get("is_short", False)),
            "profit_total_long_abs": sum(t.get("profit_abs", 0) for t in trades if not t.get("is_short", False)),
            "profit_total_short_abs": sum(t.get("profit_abs", 0) for t in trades if t.get("is_short", False)),
            "profit_mean": profit_mean,
            "profit_median": profit_median,
            "starting_balance": self.starting_balance,
            "final_balance": final_balance,
            "max_drawdown_abs": max_drawdown_abs,
            "max_drawdown_account": max_drawdown_account,
            "backtest_start_ts": backtest_start_ts,
            "backtest_end_ts": backtest_end_ts,
            "backtest_days": backtest_days,
            "trades_per_day": trades_per_day,
            **holding_stats
        }
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics when no trades exist."""
        return {
            "total_trades": 0,
            "trade_count_long": 0,
            "trade_count_short": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "winrate": 0,
            "profit_total": 0,
            "profit_total_abs": 0,
            "profit_total_long": 0,
            "profit_total_short": 0,
            "profit_total_long_abs": 0,
            "profit_total_short_abs": 0,
            "profit_mean": 0,
            "profit_median": 0,
            "starting_balance": self.starting_balance,
            "final_balance": self.starting_balance,
            "max_drawdown_abs": 0,
            "max_drawdown_account": 0,
            "backtest_start_ts": 0,
            "backtest_end_ts": 0,
            "backtest_days": 0,
            "trades_per_day": 0,
            "holding_avg": 0,
            "holding_avg_s": 0,
            "winner_holding_avg": 0,
            "winner_holding_avg_s": 0,
            "loser_holding_avg": 0,
            "loser_holding_avg_s": 0
        }
    
    def _calculate_drawdown(
        self, 
        trades: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown from trade sequence.
        
        Args:
            trades: List of trades sorted by timestamp
            
        Returns:
            Tuple of (max_drawdown_abs, max_drawdown_account)
        """
        if not trades:
            return 0.0, 0.0
        
        # Calculate cumulative profit
        cumulative = 0.0
        peak = 0.0
        max_drawdown_abs = 0.0
        
        for trade in trades:
            cumulative += trade.get("profit_abs", 0)
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown_abs:
                max_drawdown_abs = drawdown
        
        # Calculate account-based drawdown
        max_drawdown_account = max_drawdown_abs / self.starting_balance if self.starting_balance > 0 else 0
        
        return max_drawdown_abs, max_drawdown_account
    
    def _calculate_holding_stats(
        self, 
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate holding time statistics.
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary of holding time statistics
        """
        if not trades:
            return {
                "holding_avg": 0,
                "holding_avg_s": 0,
                "winner_holding_avg": 0,
                "winner_holding_avg_s": 0,
                "loser_holding_avg": 0,
                "loser_holding_avg_s": 0
            }
        
        durations = [t.get("trade_duration", 0) for t in trades]
        winner_durations = [t.get("trade_duration", 0) for t in trades if t.get("profit_ratio", 0) > 0]
        loser_durations = [t.get("trade_duration", 0) for t in trades if t.get("profit_ratio", 0) < 0]
        
        holding_avg = sum(durations) / len(durations) if durations else 0
        winner_holding_avg = sum(winner_durations) / len(winner_durations) if winner_durations else 0
        loser_holding_avg = sum(loser_durations) / len(loser_durations) if loser_durations else 0
        
        return {
            "holding_avg": holding_avg,
            "holding_avg_s": holding_avg * 60,  # Convert minutes to seconds
            "winner_holding_avg": winner_holding_avg,
            "winner_holding_avg_s": winner_holding_avg * 60,
            "loser_holding_avg": loser_holding_avg,
            "loser_holding_avg_s": loser_holding_avg * 60
        }
    
    def _build_merged_result(
        self,
        trades: List[Dict[str, Any]],
        stats: Dict[str, Any],
        results_per_pair: List[Dict[str, Any]],
        pairlist: List[str],
        strategy_name: str,
        timerange: Optional[str],
        timeframe: str,
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the complete merged result structure.
        
        Args:
            trades: Merged and sorted trades
            stats: Calculated statistics
            results_per_pair: Per-pair result summaries
            pairlist: List of trading pairs
            strategy_name: Strategy name
            timerange: Time range string
            timeframe: Timeframe string
            base_metadata: Base metadata from original results
            
        Returns:
            Complete Freqtrade-compatible result structure
        """
        # Find best and worst pairs
        best_pair = self._find_best_pair(results_per_pair)
        worst_pair = self._find_worst_pair(results_per_pair)
        
        # Build exit reason summary
        exit_reason_summary = self._build_exit_reason_summary(trades)
        
        # Build enter tag stats
        enter_tag_stats = self._build_enter_tag_stats(trades)
        
        # Get timestamps for backtest range
        backtest_start_ts = stats.get("backtest_start_ts", 0)
        backtest_end_ts = stats.get("backtest_end_ts", 0)
        
        # Convert timestamps to ISO format
        backtest_start = datetime.fromtimestamp(backtest_start_ts / 1000, tz=None).strftime("%Y-%m-%d %H:%M:%S") if backtest_start_ts else ""
        backtest_end = datetime.fromtimestamp(backtest_end_ts / 1000, tz=None).strftime("%Y-%m-%d %H:%M:%S") if backtest_end_ts else ""
        
        now_ts = int(datetime.now().timestamp())
        
        strategy_data = {
            "trades": trades,
            "locks": [],
            "best_pair": best_pair,
            "worst_pair": worst_pair,
            "results_per_pair": results_per_pair,
            "results_per_enter_tag": enter_tag_stats,
            "exit_reason_summary": exit_reason_summary,
            "mix_tag_stats": [],
            "left_open_trades": [],
            "total_trades": stats["total_trades"],
            "trade_count_long": stats["trade_count_long"],
            "trade_count_short": stats["trade_count_short"],
            "total_volume": sum(t.get("stake_amount", 0) for t in trades),
            "avg_stake_amount": sum(t.get("stake_amount", 0) for t in trades) / len(trades) if trades else 0,
            "profit_mean": stats["profit_mean"],
            "profit_median": stats["profit_median"],
            "profit_total": stats["profit_total"],
            "profit_total_long": stats["profit_total_long"],
            "profit_total_short": stats["profit_total_short"],
            "profit_total_abs": stats["profit_total_abs"],
            "profit_total_long_abs": stats["profit_total_long_abs"],
            "profit_total_short_abs": stats["profit_total_short_abs"],
            "backtest_start": backtest_start,
            "backtest_start_ts": backtest_start_ts // 1000 if backtest_start_ts else 0,
            "backtest_end": backtest_end,
            "backtest_end_ts": backtest_end_ts // 1000 if backtest_end_ts else 0,
            "backtest_days": stats["backtest_days"],
            "backtest_run_start_ts": now_ts,
            "backtest_run_end_ts": now_ts,
            "trades_per_day": stats["trades_per_day"],
            "pairlist": pairlist,
            "stake_currency": base_metadata.get("stake_currency", self.stake_currency),
            "stake_currency_decimals": base_metadata.get("stake_currency_decimals", 8),
            "starting_balance": stats["starting_balance"],
            "dry_run_wallet": base_metadata.get("dry_run_wallet", stats["starting_balance"]),
            "final_balance": stats["final_balance"],
            "max_open_trades": base_metadata.get("max_open_trades", 1),
            "max_open_trades_setting": base_metadata.get("max_open_trades_setting", 1),
            "timeframe": timeframe,
            "timeframe_detail": base_metadata.get("timeframe_detail"),
            "timerange": timerange or "",
            "enable_protections": base_metadata.get("enable_protections", False),
            "strategy_name": strategy_name,
            "stoploss": base_metadata.get("stoploss", -0.1),
            "trailing_stop": base_metadata.get("trailing_stop", False),
            "trailing_stop_positive": base_metadata.get("trailing_stop_positive"),
            "trailing_stop_positive_offset": base_metadata.get("trailing_stop_positive_offset", 0),
            "trailing_only_offset_is_reached": base_metadata.get("trailing_only_offset_is_reached", False),
            "use_custom_stoploss": base_metadata.get("use_custom_stoploss", False),
            "minimal_roi": base_metadata.get("minimal_roi", {}),
            "use_exit_signal": base_metadata.get("use_exit_signal", True),
            "exit_profit_only": base_metadata.get("exit_profit_only", False),
            "exit_profit_offset": base_metadata.get("exit_profit_offset", 0),
            "ignore_roi_if_entry_signal": base_metadata.get("ignore_roi_if_entry_signal", False),
            "trading_mode": base_metadata.get("trading_mode", "spot"),
            "margin_mode": base_metadata.get("margin_mode", ""),
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "winrate": stats["winrate"],
            "holding_avg": stats["holding_avg"],
            "holding_avg_s": stats["holding_avg_s"],
            "winner_holding_avg": stats["winner_holding_avg"],
            "winner_holding_avg_s": stats["winner_holding_avg_s"],
            "loser_holding_avg": stats["loser_holding_avg"],
            "loser_holding_avg_s": stats["loser_holding_avg_s"],
            "max_drawdown_account": stats["max_drawdown_account"],
            "max_drawdown_abs": stats["max_drawdown_abs"],
            "rejected_signals": 0,
            "timedout_entry_orders": 0,
            "timedout_exit_orders": 0,
            "canceled_trade_entries": 0,
            "canceled_entry_orders": 0,
            "replaced_entry_orders": 0
        }
        
        return {
            "strategy": {
                strategy_name: strategy_data
            },
            "strategy_comparison": [{
                "key": strategy_name,
                "trades": stats["total_trades"],
                "profit_mean": stats["profit_mean"],
                "profit_total": stats["profit_total"],
                "profit_total_abs": stats["profit_total_abs"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "draws": stats["draws"],
                "winrate": stats["winrate"],
                "max_drawdown_account": stats["max_drawdown_account"],
                "max_drawdown_abs": stats["max_drawdown_abs"]
            }]
        }
    
    def _find_best_pair(
        self, 
        results_per_pair: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the best performing pair by profit."""
        if not results_per_pair:
            return {"key": "", "profit_total_abs": 0}
        
        best = max(results_per_pair, key=lambda x: x.get("profit_total_abs", 0))
        return {"key": best.get("key", ""), "profit_total_abs": best.get("profit_total_abs", 0)}
    
    def _find_worst_pair(
        self, 
        results_per_pair: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the worst performing pair by profit."""
        if not results_per_pair:
            return {"key": "", "profit_total_abs": 0}
        
        worst = min(results_per_pair, key=lambda x: x.get("profit_total_abs", 0))
        return {"key": worst.get("key", ""), "profit_total_abs": worst.get("profit_total_abs", 0)}
    
    def _build_exit_reason_summary(
        self, 
        trades: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build summary of exit reasons."""
        exit_reasons: Dict[str, Dict[str, Any]] = {}
        
        for trade in trades:
            reason = trade.get("exit_reason", "unknown")
            if reason not in exit_reasons:
                exit_reasons[reason] = {
                    "key": reason,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "profit_total_abs": 0
                }
            
            exit_reasons[reason]["trades"] += 1
            profit = trade.get("profit_ratio", 0)
            exit_reasons[reason]["profit_total_abs"] += trade.get("profit_abs", 0)
            
            if profit > 0:
                exit_reasons[reason]["wins"] += 1
            elif profit < 0:
                exit_reasons[reason]["losses"] += 1
            else:
                exit_reasons[reason]["draws"] += 1
        
        return list(exit_reasons.values())
    
    def _build_enter_tag_stats(
        self, 
        trades: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build statistics by enter tag."""
        enter_tags: Dict[str, Dict[str, Any]] = {}
        
        for trade in trades:
            tag = trade.get("enter_tag", "")
            if tag not in enter_tags:
                enter_tags[tag] = {
                    "key": tag,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "profit_total_abs": 0
                }
            
            enter_tags[tag]["trades"] += 1
            profit = trade.get("profit_ratio", 0)
            enter_tags[tag]["profit_total_abs"] += trade.get("profit_abs", 0)
            
            if profit > 0:
                enter_tags[tag]["wins"] += 1
            elif profit < 0:
                enter_tags[tag]["losses"] += 1
            else:
                enter_tags[tag]["draws"] += 1
        
        return list(enter_tags.values())
    
    def _write_output_files(
        self,
        merged_data: Dict[str, Any],
        output_path: str,
        strategy_name: str
    ) -> None:
        """
        Write merged result to output files.
        
        Creates:
        - Main result JSON file (or ZIP if output_path ends with .zip)
        - .meta.json metadata file
        
        Args:
            merged_data: Complete merged result data
            output_path: Output file path
            strategy_name: Strategy name for metadata
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write main result file
        json_content = json.dumps(merged_data, indent=2)
        
        if output_path.endswith(".zip"):
            # Write as ZIP file
            json_filename = os.path.basename(output_path).replace(".zip", ".json")
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(json_filename, json_content)
        else:
            # Write as plain JSON
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_content)
        
        # Write .meta.json file
        meta_path = output_path.replace(".zip", ".meta.json").replace(".json", ".meta.json")
        if not meta_path.endswith(".meta.json"):
            meta_path = output_path + ".meta.json"
        
        strategy_data = merged_data.get("strategy", {}).get(strategy_name, {})
        meta_data = {
            strategy_name: {
                "run_id": f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "backtest_start_time": int(datetime.now().timestamp()),
                "timeframe": strategy_data.get("timeframe", "1m"),
                "timeframe_detail": strategy_data.get("timeframe_detail"),
                "backtest_start_ts": strategy_data.get("backtest_start_ts", 0),
                "backtest_end_ts": strategy_data.get("backtest_end_ts", 0)
            }
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f)
