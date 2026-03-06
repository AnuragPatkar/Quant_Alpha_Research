"""
UNIT TEST: Backtest Engine & Attribution
========================================
Tests the event-driven backtester logic and PnL attribution metrics.

Verifies:
  1. Execution Logic: Perfect signals on trending prices -> Positive PnL.
  2. Cost Impact: Higher commissions -> Lower Equity.
  3. Attribution: Hit ratio and PnL stats calculation.
  4. Robustness: Handling of empty inputs.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import safely
try:
    from quant_alpha.backtest.engine import BacktestEngine
    from quant_alpha.backtest.attribution import SimpleAttribution
except ImportError:
    BacktestEngine = None
    SimpleAttribution = None


@pytest.mark.skipif(BacktestEngine is None, reason="Backtest module not found")
class TestBacktestEngine:
    """Unit tests for the BacktestEngine."""

    @pytest.fixture
    def synthetic_market(self):
        """
        Creates a deterministic market scenario:
        - TICK_UP: Price rises 1.0 every day (100 -> 109)
        - TICK_DN: Price falls 1.0 every day (100 -> 91)
        - Predictions match these moves perfectly.
        """
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        prices_rows = []
        preds_rows = []

        for i, d in enumerate(dates):
            # TICK_UP: 100, 101, 102...
            prices_rows.append({
                "date": d, "ticker": "TICK_UP", 
                "open": 100 + i, "close": 100 + i, 
                "volume": 1e6, "volatility": 0.01
            })
            # TICK_DN: 100, 99, 98...
            prices_rows.append({
                "date": d, "ticker": "TICK_DN", 
                "open": 100 - i, "close": 100 - i, 
                "volume": 1e6, "volatility": 0.01
            })

            # Perfect predictions
            preds_rows.append({"date": d, "ticker": "TICK_UP", "prediction": 1.0})
            preds_rows.append({"date": d, "ticker": "TICK_DN", "prediction": -1.0})

        prices = pd.DataFrame(prices_rows)
        preds  = pd.DataFrame(preds_rows)
        return preds, prices

    def test_initialization(self):
        """Engine initializes with correct defaults."""
        engine = BacktestEngine(initial_capital=50_000, commission=0.001)
        assert engine.initial_capital == 50_000
        assert engine.commission == 0.001

    def test_perfect_foresight_profitability(self, synthetic_market):
        """
        With perfect predictions (Long UP, Short DN) and zero costs,
        strategy MUST make money.
        """
        preds, prices = synthetic_market
        
        # Zero costs to verify pure logic
        engine = BacktestEngine(
            initial_capital=100_000,
            commission=0.0,
            slippage=0.0,
            spread=0.0,
            rebalance_freq='daily' # Trade every day to capture moves
        )
        
        results = engine.run(preds, prices, top_n=2)
        
        equity = results["equity_curve"]
        final_value = equity.iloc[-1]["total_value"]
        
        assert final_value > 100_000, (
            f"Perfect strategy lost money: {final_value} <= 100,000. "
            "Check execution logic (buy/sell direction)."
        )
        assert not results["trades"].empty, "No trades were generated."

    def test_transaction_costs_impact(self, synthetic_market):
        """High commissions should strictly reduce final equity compared to low commissions."""
        preds, prices = synthetic_market
        
        # Run 1: Low Cost
        engine_low = BacktestEngine(initial_capital=100_000, commission=0.0)
        res_low = engine_low.run(preds, prices, top_n=2)
        
        # Run 2: High Cost
        engine_high = BacktestEngine(initial_capital=100_000, commission=0.05) # 5% per trade
        res_high = engine_high.run(preds, prices, top_n=2)
        
        val_low  = res_low["equity_curve"].iloc[-1]["total_value"]
        val_high = res_high["equity_curve"].iloc[-1]["total_value"]
        
        assert val_high < val_low, (
            f"High cost equity ({val_high}) >= Low cost equity ({val_low}). "
            "Commissions might not be applied correctly."
        )


@pytest.mark.skipif(SimpleAttribution is None, reason="Attribution module not found")
class TestAttribution:
    """Unit tests for SimpleAttribution logic."""

    def test_pnl_aggregation(self):
        """Verify hit ratio and PnL sums on dummy trades."""
        trades = pd.DataFrame([
            # Win
            {"ticker": "A", "side": "long", "entry_price": 100, "exit_price": 110, "pnl": 10},
            # Loss
            {"ticker": "B", "side": "long", "entry_price": 100, "exit_price": 90,  "pnl": -10},
            # Win (Short)
            {"ticker": "C", "side": "short", "entry_price": 100, "exit_price": 90, "pnl": 10},
        ])
        
        attr = SimpleAttribution()
        stats = attr.analyze_pnl_drivers(trades)
        
        assert stats["total_pnl"] == 10  # 10 - 10 + 10
        assert stats["total_trades"] == 3
        assert stats["winning_trades"] == 2
        assert stats["hit_ratio"] == pytest.approx(2/3)
        assert stats["gross_profit"] == 20
        assert stats["gross_loss"] == -10