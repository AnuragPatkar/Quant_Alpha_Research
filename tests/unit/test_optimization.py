r"""
Portfolio Optimization Validation Suite
=======================================
Validates the mathematical convergence and allocation boundaries of portfolio optimizers.

Purpose
-------
This module provides exhaustive unit tests for the platform's portfolio construction
algorithms, including Mean-Variance, Risk Parity, Kelly Criterion, and Black-Litterman.
It enforces strict checks on simplex constraints, long-only boundaries, and covariance
scaling to ensure solvers output numerically stable, fully-invested allocations.

Role in Quantitative Workflow
-----------------------------
Acts as the final safeguard before target weights are converted into discrete orders.
Ensures that optimization algorithms gracefully handle edge cases (e.g., singular
covariance matrices, missing capitalizations) without triggering execution halts.

Dependencies
------------
- **Pytest**: Orchestration of isolated test cases and data fixtures.
- **NumPy/Pandas**: Synthesis of localized covariance and return matrices.
- **Unittest.Mock**: API degradation simulation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_alpha.optimization.allocator import PortfolioAllocator
from quant_alpha.optimization.mean_variance import MeanVarianceOptimizer
from quant_alpha.optimization.risk_parity import RiskParityOptimizer
from quant_alpha.optimization.kelly_criterion import KellyCriterion
from quant_alpha.optimization.black_litterman import BlackLittermanModel

@pytest.fixture(scope="module")
def market_data():
    r"""
    Provisions a localized 3-asset market environment to test capital allocation bounds.

    Assets:
    - **A**: High-return, low-volatility (Alpha generator). Block-diagonal covariance.
    - **B**: Low-return, high-volatility (Junk asset).
    - **C**: Moderate return and volatility (Market baseline).

    Args:
        None

    Returns:
        tuple: A 2-element tuple containing:
            - pd.Series: The expected annualized returns ($\mu$).
            - pd.DataFrame: The $3 \times 3$ Covariance matrix ($\Sigma$).
    """
    tickers = ["A", "B", "C"]
    mu = pd.Series([0.15, 0.02, 0.08], index=tickers)
    cov_data = [
        [0.04, 0.00, 0.00],   
        [0.00, 0.09, 0.04],   
        [0.00, 0.04, 0.06],   
    ]
    cov = pd.DataFrame(cov_data, index=tickers, columns=tickers)
    return mu, cov

@pytest.fixture(scope="module")
def large_market_data():
    """
    Generates a 10-asset universe to evaluate matrix scaling and numerical stability.

    Constructs linearly increasing expected returns bound to a positive semi-definite 
    covariance matrix derived via $A^T A$.

    Args:
        None

    Returns:
        tuple: A 2-element tuple containing the expected returns and covariance matrix.
    """
    np.random.seed(0)
    n = 10
    tickers = [f"S{i}" for i in range(n)]
    mu = pd.Series(np.linspace(0.01, 0.19, n), index=tickers)
    A = np.random.randn(n, n) * 0.02
    cov_raw = A.T @ A + np.diag(np.linspace(0.01, 0.09, n))
    cov = pd.DataFrame(cov_raw, index=tickers, columns=tickers)
    return mu, cov

def _w(weights_dict, index):
    """
    Transforms a dictionary of allocations into a strictly aligned pandas Series.

    Args:
        weights_dict (dict): Optimization output mapping tickers to float weights.
        index (pd.Index): The target index for structural alignment.

    Returns:
        pd.Series: A dimensionally verified series, filling missing allocations with 0.0.
    """
    return pd.Series(weights_dict).reindex(index).fillna(0.0)

def _assert_valid_portfolio(weights: pd.Series, *, tol: float = 1e-4):
    """
    Asserts foundational simplex boundaries governing long-only portfolios.

    Args:
        weights (pd.Series): The generated allocation weights.
        tol (float, optional): The absolute floating-point tolerance for unity summation. 
            Defaults to 1e-4.

    Returns:
        None

    Raises:
        AssertionError: If weights do not sum to 1.0 or if negative allocations breach epsilon.
    """
    assert weights.sum() == pytest.approx(1.0, abs=tol), (
        f"Weights sum to {weights.sum():.6f}, expected 1.0"
    )
    # Enforces strictly negative limits down to -1e-8 to allow for negligible solver artifacts
    assert (weights >= -1e-8).all(), (  
        f"Long-only violated: {weights[weights < -1e-8].to_dict()}"
    )

def _compute_port_vol(
    allocator: PortfolioAllocator,
    mu: pd.Series,
    cov: pd.DataFrame,
) -> float:
    """
    Calculates expected portfolio volatility for a specific allocator configuration.

    Args:
        allocator (PortfolioAllocator): The instantiated configuration orchestrator.
        mu (pd.Series): Expected return matrix.
        cov (pd.DataFrame): Asset covariance matrix.

    Returns:
        float: The expected annualized standard deviation of the constructed portfolio.
    """
    weights = _w(
        allocator.allocate(mu.to_dict(), cov, constraints={"max_weight": 1.0}),
        mu.index,
    )
    return float(np.sqrt(weights.dot(cov).dot(weights)))

class TestMeanVariance:
    """
    Validation suite for standard Markowitz Quadratic Programming formulations.
    """

    def test_dominant_asset_gets_max_weight(self, market_data):
        """
        Validates that alpha-dominant assets naturally capture execution limits.

        Asserts that an asset exhibiting a superior structural Sharpe Ratio mathematically 
        dominates the optimized weight vector. Explicitly overriding `max_weight=1.0` 
        prevents artificial cardinality caps from forcing equal distribution.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator(method="mean_variance", risk_aversion=5.0).allocate(
                mu.to_dict(), cov, constraints={"max_weight": 1.0}
            ),
            mu.index,
        )

        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["B"], "A should dominate junk asset B"
        # Enforces a minimum 5-percentage-point margin to definitively prove optimization hierarchy
        assert weights["A"] > weights["C"] + 0.05, (  
            f"A ({weights['A']:.4f}) did not dominate C ({weights['C']:.4f}) by ≥5pp"
        )
        assert weights["B"] < 0.10, "Junk asset B should have near-zero weight"

    def test_max_weight_constraint_respected(self, market_data):
        """
        Validates the strict enforcement of absolute concentration ceilings.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator(method="mean_variance").allocate(
                mu.to_dict(), cov, constraints={"max_weight": 0.40}
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert (weights <= 0.40 + 1e-6).all(), (
            f"max_weight=0.40 violated: {weights[weights > 0.40 + 1e-6].to_dict()}"
        )

    def test_max_sharpe_portfolio_exceeds_lower_bound(self, market_data):
        r"""
        Evaluates the solver's ability to approximate the global Sharpe optimum via SOCP.

        Mathematical derivation confirms Asset A produces an isolated Sharpe of 0.75. 
        The optimizer must successfully linearize the objective space via the 
        Charnes-Cooper transformation to cross the 0.70 boundary regardless of the 
        underlying cone solver executing (e.g., ECOS vs SCS).

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            MeanVarianceOptimizer().solve_max_sharpe(
                mu.to_dict(), cov, risk_free_rate=0.0
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)

        port_vol = np.sqrt(weights.dot(cov).dot(weights))
        # Precludes zero-variance division errors
        assert port_vol > 1e-10, f"Degenerate portfolio vol ({port_vol:.2e})"  

        port_sharpe = weights.dot(mu) / port_vol
        assert port_sharpe >= 0.70, (
            f"Max-Sharpe Sharpe ({port_sharpe:.4f}) < 0.70"
        )

    def test_higher_risk_aversion_lowers_portfolio_volatility(self, market_data):
        r"""
        Verifies that increasing the structural risk aversion penalty ($\lambda$) 
        mathematically reduces geometric portfolio variance.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        vol_low  = _compute_port_vol(PortfolioAllocator("mean_variance", risk_aversion=0.1),  mu, cov)
        vol_high = _compute_port_vol(PortfolioAllocator("mean_variance", risk_aversion=10.0), mu, cov)
        assert vol_high < vol_low, (
            f"Higher risk_aversion should reduce vol: ra=0.1→{vol_low:.4f}, ra=10→{vol_high:.4f}"
        )

    def test_singular_covariance_does_not_crash(self):
        """
        Validates pipeline resilience when facing degenerate or perfectly correlated state matrices.

        Args:
            None

        Returns:
            None
        """
        tickers = ["X", "Y"]
        mu  = pd.Series([0.10, 0.10], index=tickers)
        cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=tickers, columns=tickers)
        try:
            weights = _w(
                PortfolioAllocator("mean_variance").allocate(mu.to_dict(), cov),
                mu.index,
            )
        except Exception as e:
            pytest.fail(f"Crashed on singular matrix: {type(e).__name__}: {e}")

        assert weights.sum() == pytest.approx(1.0, abs=1e-4)
        assert not weights.isna().any()

    def test_nan_in_expected_returns_handled_gracefully(self, market_data):
        """
        Ensures missing structural inputs trigger elegant fallbacks rather than runtime faults.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        mu = mu.copy()   
        mu["B"] = np.nan
        try:
            weights = _w(
                PortfolioAllocator("mean_variance").allocate(mu.to_dict(), cov),
                mu.index,
            )
            assert weights.sum() == pytest.approx(1.0, abs=1e-4)
        except Exception as e:
            pytest.fail(f"Unexpected {type(e).__name__} on NaN input: {e}")

    def test_single_asset_gets_full_allocation(self):
        """
        Asserts discrete boundaries automatically map unified allocations to lone survivors.

        Args:
            None

        Returns:
            None
        """
        mu  = pd.Series([0.10], index=["SOLO"])
        cov = pd.DataFrame([[0.04]], index=["SOLO"], columns=["SOLO"])
        try:
            weights = _w(
                PortfolioAllocator("mean_variance").allocate(mu.to_dict(), cov),
                mu.index,
            )
        except Exception as e:
            pytest.fail(f"Crashed on single-asset: {type(e).__name__}: {e}")
        assert weights["SOLO"] == pytest.approx(1.0, abs=1e-4)

    def test_empty_input_returns_empty_dict(self):
        """
        Validates completely hollow execution regimes collapse securely to empty dictionaries.

        Args:
            None

        Returns:
            None
        """
        result = PortfolioAllocator("mean_variance").allocate({}, pd.DataFrame())
        assert result == {}

    def test_output_is_dict_with_valid_keys(self, market_data):
        """
        Asserts strict type checking and dictionary dimensional compliance.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        result = PortfolioAllocator("mean_variance").allocate(mu.to_dict(), cov)
        assert isinstance(result, dict)
        assert set(result.keys()).issubset(set(mu.index))

    def test_max_weight_floor_logic(self, market_data):
        r"""
        Evaluates automated constraint relaxation to guarantee mathematical feasibility.

        If a user specifies a target ceiling that is mathematically impossible 
        to achieve while maintaining a $100\%$ fully-invested portfolio ($w_{max} < 1/N$), 
        the solver must override the input to the infeasibility floor to prevent crashes.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        n = len(mu)
        allocator = PortfolioAllocator("mean_variance")
        weights = _w(
            allocator.allocate(mu.to_dict(), cov, constraints={"max_weight": 0.1}),
            mu.index
        )
        _assert_valid_portfolio(weights)
        assert (weights > 0.1).any()

class TestRiskParity:
    r"""
    Validation suite for Equal Risk Contribution algorithms based on Spinu's Log-Barrier formulation.
    """

    def test_weight_ordering_matches_inverse_vol(self, market_data):
        """
        Validates fundamental risk-contribution proportionality.

        Ensures that strictly lower-volatility assets are mathematically assigned 
        higher nominal weights to balance Marginal Risk Contributions (MRC). The 
        Spinu formulation explicitly prevents block-diagonal matrices from trapping 
        solvers at $0.0$ bounds.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["C"], (
            f"w_A={weights['A']:.4f} should exceed w_C={weights['C']:.4f} (vol A=0.20 < C=0.245)"
        )
        assert weights["C"] > weights["B"], (
            f"w_C={weights['C']:.4f} should exceed w_B={weights['B']:.4f} (vol C=0.245 < B=0.30)"
        )

    def test_equal_risk_contributions(self, market_data):
        r"""
        Evaluates cross-sectional equality of Marginal Risk Contributions.

        Ensures that $MRC_i = \frac{\partial \sigma_p}{\partial w_i}$ remains 
        consistent across all executing constraints within acceptable tolerances.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
            mu.index,
        )
        port_vol = np.sqrt(weights.dot(cov).dot(weights))
        assert port_vol > 1e-10, "Degenerate portfolio vol"  

        mrc = weights * (cov.dot(weights)) / port_vol
        assert np.allclose(mrc, mrc.mean(), rtol=0.15), (
            f"MRC not equal across assets: {dict(zip(mu.index, mrc.round(6)))}"
        )

    def test_block_diagonal_asset_has_nonzero_weight(self, market_data):
        """
        Verifies non-zero boundaries on completely uncorrelated assets.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
            mu.index,
        )
        assert weights["A"] > 0.01, (
            f"Block-diagonal asset A collapsed to w_A={weights['A']:.6f}. "
            "Spinu formulation should prevent this."
        )

    def test_custom_risk_budgets(self, market_data):
        """
        Verifies that explicit budget configurations manipulate underlying weight distributions.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        optimizer = RiskParityOptimizer(target_risk={"A": 0.6, "B": 0.2, "C": 0.2})
        weights = _w(optimizer.optimize(cov, list(mu.index)), mu.index)

        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["B"], "Higher budget for A must yield higher weight than B"
        assert weights["A"] > weights["C"], "Higher budget for A must yield higher weight than C"

    def test_large_universe_all_assets_get_positive_weight(self, large_market_data):
        """
        Validates the strict positive adherence of the Spinu logarithmic barrier.

        Args:
            large_market_data (tuple): Synthesized scaled test matrix.

        Returns:
            None
        """
        mu, cov = large_market_data
        weights = _w(
            PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert (weights > 0).all(), "All assets must have positive weight in risk parity"

    def test_fallback_on_solver_failure(self, market_data):
        """
        Validates graceful algorithmic degradation upon SciPy convergence failure.

        If the L-BFGS-B gradient solver halts, the system must deterministically 
        fall back to a stable Inverse Volatility heuristic rather than crashing the pipeline.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        
        mock_res = type('MockResult', (), {'success': False, 'message': 'Mock failure'})()
        
        with patch("quant_alpha.optimization.risk_parity.minimize", return_value=mock_res):
            weights = _w(
                PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
                mu.index,
            )
        
        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["B"]

    def test_optimize_subset_of_tickers(self, market_data):
        """
        Ensures matrix filtering correctly isolates specific ticker boundaries.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        mu_subset = mu[["A", "B"]]
        
        weights = _w(
            PortfolioAllocator("risk_parity").allocate(mu_subset.to_dict(), cov),
            mu_subset.index,
        )
        
        assert len(weights) == 2
        assert "C" not in weights
        assert weights.sum() == pytest.approx(1.0)

    def test_optimize_empty_tickers(self, market_data):
        """
        Verifies empty matrix dimensions bypass evaluation loops dynamically.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        _, cov = market_data
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(cov, [])
        assert result == {}

class TestKellyCriterion:
    r"""
    Validation suite for Geometric Growth Maximization via the Kelly Criterion constraints.
    """

    def test_higher_return_asset_dominates(self, market_data):
        """
        Asserts proportional dominance for vectors with high excess return/variance ratios.

        Assets failing to bridge the risk-free rate hurdle should be completely 
        omitted post-normalization.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator("kelly").allocate(
                mu.to_dict(), cov, risk_free_rate=0.04
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["C"], (
            f"Kelly: w_A={weights['A']:.4f} should exceed w_C={weights['C']:.4f}"
        )
        assert weights["B"] < 0.05, (
            f"Kelly: B has negative excess return — should be near-zero, got {weights['B']:.4f}"
        )

    def test_full_kelly_more_concentrated_than_half_kelly(self, market_data):
        """
        Verifies risk reduction mechanics using Fractional Kelly scaling logic.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        w_full = _w(
            PortfolioAllocator("kelly", fraction=1.0).allocate(
                mu.to_dict(), cov, risk_free_rate=0.04
            ),
            mu.index,
        )
        w_half = _w(
            PortfolioAllocator("kelly", fraction=0.5).allocate(
                mu.to_dict(), cov, risk_free_rate=0.04
            ),
            mu.index,
        )
        _assert_valid_portfolio(w_full)
        _assert_valid_portfolio(w_half)
        assert w_full.max() >= w_half.max() - 1e-4, (
            "Full Kelly should be at least as concentrated as half-Kelly"
        )

    def test_zero_risk_free_rate_produces_valid_portfolio(self, market_data):
        """
        Validates execution bounding for perfectly elastic zero-yield benchmark configurations.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator("kelly").allocate(
                mu.to_dict(), cov, risk_free_rate=0.0
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert (weights >= 0).all()

    def test_all_negative_excess_returns_triggers_fallback(self, market_data):
        """
        Evaluates the equal-weight fallback mechanism during hostile execution conditions.

        When all generated expected returns are mathematically inferior to the prevailing 
        risk-free benchmark, the optimal mathematical solution is completely cash-based. 
        Since the engine enforces 1.0 gross exposure, this must explicitly trigger 
        a maximum entropy equal-weight distribution.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        result = KellyCriterion(fraction=0.5).calculate_portfolio(
            mu.to_dict(), cov, risk_free_rate=0.99
        )
        weights = _w(result, mu.index)
        assert weights.sum() == pytest.approx(1.0, abs=1e-4), (
            "Kelly must return a fully-invested equal-weight fallback "
            "when all excess returns are negative (rf=0.99)"
        )
        assert (weights > 0).all(), "All assets must appear in the equal-weight fallback"

    def test_output_is_dict(self, market_data):
        """
        Asserts strict type checking for the returned portfolio schema.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        result = PortfolioAllocator("kelly").allocate(
            mu.to_dict(), cov, risk_free_rate=0.04
        )
        assert isinstance(result, dict)

    def test_heuristic_solver_fallback(self, market_data):
        """
        Tests the explicit matrix inversion pathway bypassing quadratic programming parameters.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        allocator = PortfolioAllocator("kelly", fraction=0.5, use_solver=False)
        weights = _w(
            allocator.allocate(mu.to_dict(), cov, risk_free_rate=0.04),
            mu.index
        )
        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["C"]
        assert weights["B"] < 0.05

class TestBlackLitterman:
    r"""
    Validation suite for the Bayesian Black-Litterman inference and blending logic.
    """

    @pytest.fixture
    def bl_inputs(self, market_data):
        """
        Injects structured market capitalizations to formulate prior distribution bounds.

        Args:
            market_data (tuple): Foundational execution matrices.

        Returns:
            tuple: Expanded execution parameters mapping market_caps.
        """
        mu, cov = market_data
        market_caps = {"A": 500e9, "B": 50e9, "C": 250e9}
        return mu, cov, market_caps

    def test_valid_portfolio_with_all_inputs(self, bl_inputs):
        """
        Validates successful execution loop generating standard prior geometries.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        weights = _w(
            PortfolioAllocator("black_litterman").allocate(
                mu.to_dict(), cov, market_caps=market_caps
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)

    def test_high_confidence_tilts_toward_high_return_asset(self, bl_inputs):
        """
        Ensures heavy weighting of explicit model-derived views overshadows market prior distributions.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        weights = _w(
            PortfolioAllocator("black_litterman").allocate(
                mu.to_dict(), cov,
                market_caps=market_caps,
                confidence_level=0.99,
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert weights["A"] > weights["B"], (
            "High-confidence BL must tilt toward high-return Asset A"
        )

    def test_low_confidence_stays_near_market_cap_prior(self, bl_inputs):
        """
        Verifies near-zero uncertainty convergence anchors strictly toward standard market weighting.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        weights = _w(
            PortfolioAllocator("black_litterman").allocate(
                mu.to_dict(), cov,
                market_caps=market_caps,
                confidence_level=0.01,
            ),
            mu.index,
        )
        _assert_valid_portfolio(weights)
        assert weights["A"] == weights.max(), (
            "Near-zero confidence: A should be the largest weight (market-cap prior)"
        )

    def test_missing_market_caps_returns_valid_fallback(self, market_data):
        """
        Ensures fallback protocols bypass runtime exceptions when structural vectors are missing.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        try:
            weights = _w(
                PortfolioAllocator("black_litterman").allocate(mu.to_dict(), cov),
                mu.index,
            )
            assert weights.sum() == pytest.approx(1.0, abs=1e-4)
        except Exception as e:
            pytest.fail(f"BL crashed without market_caps: {type(e).__name__}: {e}")

    def test_tau_sensitivity_produces_different_portfolios(self, bl_inputs):
        """
        Verifies mathematical sensitivity tracking relative to $\tau$ hyperparameter alterations.

        Confirms that modifying the uncertainty scalar significantly impacts the posterior distribution 
        and final allocation vector, proving that the blending logic is active.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        w_tight = _w(
            PortfolioAllocator("black_litterman", tau=0.001).allocate(
                mu.to_dict(), cov, market_caps=market_caps
            ),
            mu.index,
        )
        w_loose = _w(
            PortfolioAllocator("black_litterman", tau=50.0).allocate(
                mu.to_dict(), cov, market_caps=market_caps
            ),
            mu.index,
        )
        max_diff = float(np.max(np.abs(w_tight.values - w_loose.values)))
        assert not np.allclose(w_tight.values, w_loose.values, atol=0.05), (
            f"tau=0.001 and tau=50.0 should produce portfolios differing by >5pp. "
            f"Got max_diff={max_diff:.4f}. "
            "Check BUG-BL-02: MVO must use prior Sigma, not Sigma_post."
        )

    def test_all_tau_values_produce_valid_portfolios(self, bl_inputs):
        """
        Asserts complete stability across the entire spectrum of permissible scalar weights.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        for tau in [0.01, 0.05, 0.50, 0.99]:
            weights = _w(
                PortfolioAllocator("black_litterman", tau=tau).allocate(
                    mu.to_dict(), cov, market_caps=market_caps
                ),
                mu.index,
            )
            _assert_valid_portfolio(weights)

    def test_output_is_dict_with_valid_keys(self, bl_inputs):
        """
        Evaluates deterministic map boundaries matching standard interface constraints.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        result = PortfolioAllocator("black_litterman").allocate(
            mu.to_dict(), cov, market_caps=market_caps
        )
        assert isinstance(result, dict)
        assert set(result.keys()).issubset(set(mu.index))

    def test_linalg_error_fallback(self, bl_inputs):
        """
        Tests fallback behavior during structurally impossible matrix inversions.

        Args:
            bl_inputs (tuple): Specific fixture mapping.

        Returns:
            None
        """
        mu, cov, market_caps = bl_inputs
        
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular matrix")):
            weights = _w(
                PortfolioAllocator("black_litterman").allocate(
                    mu.to_dict(), cov, market_caps=market_caps
                ),
                mu.index,
            )
        _assert_valid_portfolio(weights)

class TestPortfolioConstraints:
    """
    Validation suite verifying algorithmic enforcement of post-optimization heuristics.
    """

    @pytest.fixture
    def raw_weights(self):
        """
        Provisions intentionally malformed output distributions to verify constraint clamping logic.

        Args:
            None

        Returns:
            pd.Series: Synthetic constraint-violating array.
        """
        return pd.Series({"A": 0.60, "B": 0.25, "C": 0.145, "D": 0.005})

    def test_max_weight_cap_enforced(self, raw_weights):
        """
        Evaluates maximum allocation constraints against non-compliant arrays.

        Iterative Proportional Fitting is expected to correctly bleed excess 
        concentration to minor allocations symmetrically.

        Args:
            raw_weights (pd.Series): The invalid weight distributions.

        Returns:
            None
        """
        from quant_alpha.optimization.constraints import PortfolioConstraints

        constrained = _w(
            PortfolioConstraints(max_weight=0.40).apply(raw_weights.to_dict()),
            raw_weights.index,
        )
        assert constrained.sum() == pytest.approx(1.0, abs=1e-4)
        assert (constrained <= 0.40 + 1e-6).all(), (
            f"max_weight=0.40 violated: {constrained[constrained > 0.40 + 1e-6].to_dict()}"
        )

    def test_min_weight_floor_removes_tiny_positions(self, raw_weights):
        """
        Tests dynamic cardinality reduction thresholds.

        Args:
            raw_weights (pd.Series): The invalid weight distributions.

        Returns:
            None
        """
        from quant_alpha.optimization.constraints import PortfolioConstraints

        constrained = _w(
            PortfolioConstraints(min_weight=0.01).apply(raw_weights.to_dict()),
            raw_weights.index,
        )
        assert constrained["D"] == pytest.approx(0.0), (
            "D (0.5%) must be removed by min_weight=0.01"
        )
        assert constrained.sum() == pytest.approx(1.0, abs=1e-4)

    def test_sector_limit_enforced(self, raw_weights):
        """
        Assures aggregate bucket limits effectively truncate grouped assets to explicit ceilings.

        Args:
            raw_weights (pd.Series): The invalid weight distributions.

        Returns:
            None
        """
        from quant_alpha.optimization.constraints import PortfolioConstraints

        sector_map = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance"}
        constrained = _w(
            PortfolioConstraints(
                sector_limits={"Tech": 0.50}, sector_map=sector_map
            ).apply(raw_weights.to_dict()),
            raw_weights.index,
        )
        tech_total = constrained["A"] + constrained["B"]
        assert tech_total <= 0.50 + 1e-6, (
            f"Tech sector weight {tech_total:.4f} exceeds limit 0.50"
        )
        assert constrained.sum() == pytest.approx(1.0, abs=1e-4)

    def test_already_valid_portfolio_is_unchanged(self):
        """
        Confirms idempotent bypass for already valid distributions reducing unnecessary compute overhead.

        Args:
            None

        Returns:
            None
        """
        from quant_alpha.optimization.constraints import PortfolioConstraints

        equal_w = {"A": 0.33, "B": 0.33, "C": 0.34}
        constrained = PortfolioConstraints(max_weight=1.0, min_weight=0.0).apply(equal_w)
        for ticker, w in equal_w.items():
            assert constrained.get(ticker, 0.0) == pytest.approx(w, abs=1e-4)

    def test_unmapped_sector_handling(self, raw_weights):
        """
        Tests resilient resolution for universe constituents omitting valid metadata keys.

        Args:
            raw_weights (pd.Series): The invalid weight distributions.

        Returns:
            None
        """
        from quant_alpha.optimization.constraints import PortfolioConstraints

        sector_map = {"A": "Tech", "B": "Tech"} 
        constrained = _w(
            PortfolioConstraints(
                sector_limits={"Tech": 0.50}, sector_map=sector_map
            ).apply(raw_weights.to_dict()),
            raw_weights.index,
        )
        assert constrained["A"] + constrained["B"] <= 0.50 + 1e-6
        assert constrained["C"] > 0.145  # C should receive some redistributed weight

class TestPortfolioAllocatorFacade:
    """
    Verification block strictly modeling the structural integration facade patterns.
    """

    def test_unknown_method_raises_value_error(self):
        """
        Asserts instantiation immediately aborts upon explicit logic misconfigurations.

        Args:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Unknown optimization method"):
            PortfolioAllocator(method="deep_learning_magic")

    def test_equal_weight_fallback_on_mismatched_covariance(self, market_data):
        """
        Confirms systemic failover behavior against disjoint or entirely missing data bounds.

        When covariance evaluation matrices lack unified intersection against 
        the target expected return vectors, defensive allocation protocols must
        strictly enforce completely flat positional structures rather than halting explicitly.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, _ = market_data
        bad_cov = pd.DataFrame([[0.04]], index=["Z"], columns=["Z"])

        result = PortfolioAllocator("mean_variance").allocate(mu.to_dict(), bad_cov)
        weights = _w(result, mu.index)

        assert weights.sum() == pytest.approx(1.0, abs=1e-4), (
            "Allocator must fall back to equal weight when optimizer returns empty dict. "
            "Fix: in allocator.py, after calling optimizer, check if result is empty "
            "and substitute {t: 1/n for t in expected_returns.keys()}."
        )

    @pytest.mark.parametrize(
        "method,extra_kwargs",
        [
            ("mean_variance",    {}),
            ("risk_parity",      {}),
            ("kelly",            {"risk_free_rate": 0.04}),
            ("black_litterman",  {"market_caps": {"A": 500e9, "B": 50e9, "C": 250e9}}),
        ],
    )
    def test_all_methods_return_valid_portfolio(self, method, extra_kwargs, market_data):
        """
        Runs global sanity checks mapping universal compliance to discrete solver boundaries.

        Args:
            method (str): Parametrized identifier determining routing path.
            extra_kwargs (dict): Parametrized inputs governing distinct execution loops.
            market_data (tuple): Standard localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        weights = _w(
            PortfolioAllocator(method=method).allocate(mu.to_dict(), cov, **extra_kwargs),
            mu.index,
        )
        _assert_valid_portfolio(weights)

    def test_risk_aversion_kwarg_affects_output(self, market_data):
        """
        Verifies correct topological injection of secondary hyperparameters through the Facade.

        Args:
            market_data (tuple): Standard localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        w_agg = _w(
            PortfolioAllocator("mean_variance", risk_aversion=0.01).allocate(
                mu.to_dict(), cov, constraints={"max_weight": 1.0}
            ),
            mu.index,
        )
        w_con = _w(
            PortfolioAllocator("mean_variance", risk_aversion=100.0).allocate(
                mu.to_dict(), cov, constraints={"max_weight": 1.0}
            ),
            mu.index,
        )
        assert not np.allclose(w_agg.values, w_con.values, atol=0.02), (
            "risk_aversion kwarg has no effect — likely not threaded through"
        )

    def test_tau_kwarg_affects_black_litterman_output(self, market_data):
        """
        Confirms uncertainty tuning inputs accurately percolate to the statistical prior module.

        Args:
            market_data (tuple): Standard localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        market_caps = {"A": 500e9, "B": 50e9, "C": 250e9}

        w_tight = _w(
            PortfolioAllocator("black_litterman", tau=0.001).allocate(
                mu.to_dict(), cov, market_caps=market_caps
            ),
            mu.index,
        )
        w_loose = _w(
            PortfolioAllocator("black_litterman", tau=50.0).allocate(
                mu.to_dict(), cov, market_caps=market_caps
            ),
            mu.index,
        )
        max_diff = float(np.max(np.abs(w_tight.values - w_loose.values)))
        assert not np.allclose(w_tight.values, w_loose.values, atol=0.05), (
            f"tau kwarg has no effect — check BUG-BL-02 in black_litterman.py. "
            f"Got max_diff={max_diff:.4f}, expected >0.05."
        )

class TestCrossMethodProperties:
    """
    Verification suite enforcing theoretical constraints directly comparing distinct optimization approaches.
    """

    def test_risk_parity_more_balanced_than_mean_variance(self, market_data):
        """
        Verifies statistical diversification theorems inherent to ERC construction vectors.

        Risk parity diversifies by equalizing risk contribution, meaning it mathematically 
        must produce a more structurally balanced portfolio array than pure MVO on a 
        universe with one distinctly dominant asset.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        w_mv = _w(
            PortfolioAllocator("mean_variance", risk_aversion=1.0).allocate(
                mu.to_dict(), cov, constraints={"max_weight": 1.0}
            ),
            mu.index,
        )
        w_rp = _w(
            PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
            mu.index,
        )
        assert w_rp.max() < w_mv.max(), (
            f"Risk parity should be more balanced: max(RP)={w_rp.max():.4f} vs max(MV)={w_mv.max():.4f}"
        )

    def test_kelly_more_aggressive_than_risk_parity(self, market_data):
        """
        Assures relative concentration profiles between distinct solver bounds.

        Kelly's objective explicitly maximizes geometric drift, meaning it must inherently 
        generate a more concentrated vector array than the flat ERC implementation.

        Args:
            market_data (tuple): Injected localized environment matrices.

        Returns:
            None
        """
        mu, cov = market_data
        w_kelly = _w(
            PortfolioAllocator("kelly").allocate(mu.to_dict(), cov, risk_free_rate=0.04),
            mu.index,
        )
        w_rp = _w(
            PortfolioAllocator("risk_parity").allocate(mu.to_dict(), cov),
            mu.index,
        )
        assert w_kelly.max() >= w_rp.max() - 0.05, (
            f"Kelly should be at least as concentrated as RP: "
            f"max(Kelly)={w_kelly.max():.4f}, max(RP)={w_rp.max():.4f}"
        )

    def test_all_methods_handle_large_universe(self, large_market_data):
        """
        Evaluates scale and speed limits across solver configurations iteratively.

        Args:
            large_market_data (tuple): Synthesized scaled test matrix.

        Returns:
            None
        """
        mu, cov = large_market_data
        market_caps = {t: float(i + 1) * 50e9 for i, t in enumerate(mu.index)}

        cases = {
            "mean_variance":   {},
            "risk_parity":     {},
            "kelly":           {"risk_free_rate": 0.04},
            "black_litterman": {"market_caps": market_caps},
        }
        for method, extra in cases.items():
            weights = _w(
                PortfolioAllocator(method=method).allocate(mu.to_dict(), cov, **extra),
                mu.index,
            )
            _assert_valid_portfolio(weights)