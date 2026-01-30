"""
Performance Charts & Visualization
==================================
Professional static plots using matplotlib and seaborn.

Classes:
    - PerformancePlotter: Equity curves, returns, drawdowns
    - FactorPlotter: Feature importance, factor analysis
    - RiskPlotter: Risk metrics, VaR, stress testing

Output Formats:
    - PNG, JPG, SVG, PDF
    - Display in Jupyter notebooks

Author: Senior Quant Team
Last Updated: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
import logging

# Setup
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


# =============================================================================
# UTILITIES
# =============================================================================

def safe_divide(
    a: Union[float, np.ndarray, pd.Series],
    b: Union[float, np.ndarray, pd.Series],
    default: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """Safe division handling zero and NaN."""
    if isinstance(b, (pd.Series, np.ndarray)):
        result = np.where(np.abs(b) > 1e-10, a / b, default)
        if isinstance(a, pd.Series):
            return pd.Series(result, index=a.index)
        return result
    else:
        if abs(b) < 1e-10:
            return default
        return a / b


def validate_series(
    series: pd.Series,
    name: str = "series",
    min_length: int = 2
) -> pd.Series:
    """Validate and clean series data."""
    if series is None or len(series) == 0:
        raise ValueError(f"{name} is empty")
    
    series_clean = series.dropna()
    
    if len(series_clean) < min_length:
        raise ValueError(f"{name} has insufficient data (< {min_length} points)")
    
    return series_clean


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str = "dataframe"
) -> pd.DataFrame:
    """Validate dataframe has required columns."""
    if df is None or len(df) == 0:
        raise ValueError(f"{name} is empty")
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    
    return df


# =============================================================================
# PERFORMANCE PLOTTER
# =============================================================================

class PerformancePlotter:
    """
    Performance visualization for backtest results.
    
    Generates:
        - Equity curves with drawdown
        - Returns distribution analysis
        - Rolling metrics
        - Monthly heatmaps
        - Trade analysis
    
    Example:
        >>> plotter = PerformancePlotter()
        >>> fig = plotter.plot_equity_curve(equity_curve)
        >>> fig.savefig('equity.png')
        
        # With custom currency
        >>> plotter = PerformancePlotter(currency_symbol='₹', indian_format=True)
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        style: str = 'default',
        dpi: int = 150,
        currency_symbol: str = '$',
        indian_format: bool = False
    ):
        """
        Initialize plotter.
        
        Args:
            figsize: Default figure size (width, height)
            style: Plot style ('default', 'dark', 'minimal')
            dpi: Resolution for saved figures
            currency_symbol: Currency symbol to use
            indian_format: Use Indian numbering (Lakhs, Crores)
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.currency_symbol = currency_symbol
        self.indian_format = indian_format
        self._setup_colors()
    
    def _setup_colors(self) -> None:
        """Setup color scheme based on style."""
        if self.style == 'dark':
            self.colors = {
                'primary': '#00D4AA',
                'secondary': '#FF6B6B',
                'positive': '#00D4AA',
                'negative': '#FF6B6B',
                'warning': '#FFD93D',
                'neutral': '#888888',
                'benchmark': '#FFA500',
                'fill_alpha': 0.3,
                'grid_color': '#444444',
                'text': '#FFFFFF',
                'background': '#1A1A2E'
            }
        else:
            self.colors = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'positive': '#28A745',
                'negative': '#DC3545',
                'warning': '#FFC107',
                'neutral': '#6C757D',
                'benchmark': '#FF9800',
                'fill_alpha': 0.2,
                'grid_color': '#E0E0E0',
                'text': '#212529',
                'background': '#FFFFFF'
            }
    
    # =========================================================================
    # CURRENCY FORMATTING
    # =========================================================================
    
    def _format_currency(self, x: float, pos: int = 0) -> str:
        """Format number as currency (Western or Indian style)."""
        if self.indian_format:
            # Indian format: Lakhs (1e5) and Crores (1e7)
            if abs(x) >= 1e7:
                return f'{self.currency_symbol}{x/1e7:.1f}Cr'
            elif abs(x) >= 1e5:
                return f'{self.currency_symbol}{x/1e5:.1f}L'
            elif abs(x) >= 1e3:
                return f'{self.currency_symbol}{x/1e3:.0f}K'
            else:
                return f'{self.currency_symbol}{x:.0f}'
        else:
            # Western format: K, M, B, T
            if abs(x) >= 1e12:
                return f'{self.currency_symbol}{x/1e12:.1f}T'
            elif abs(x) >= 1e9:
                return f'{self.currency_symbol}{x/1e9:.1f}B'
            elif abs(x) >= 1e6:
                return f'{self.currency_symbol}{x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{self.currency_symbol}{x/1e3:.0f}K'
            else:
                return f'{self.currency_symbol}{x:.0f}'
    
    # =========================================================================
    # CALCULATION HELPERS
    # =========================================================================
    
    def _calculate_cagr(self, equity_curve: pd.Series) -> float:
        """Calculate CAGR from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if n_years <= 0:
            return 0.0
        return (total_return ** (1 / n_years) - 1) * 100
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = equity_curve.expanding(min_periods=1).max()
        return (equity_curve - peak) / peak
    
    def _max_dd_from_returns(self, returns: pd.Series) -> float:
        """Calculate max drawdown from returns series."""
        if len(returns) == 0:
            return 0.0
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()
    
    def _format_date_axis(self, ax: plt.Axes) -> None:
        """Format x-axis for dates."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _save_figure(self, fig: plt.Figure, path: str) -> None:
        """Save figure to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=self.dpi,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none'
        )
        logger.info(f"Saved plot: {save_path}")
        print(f"✅ Saved: {path}")
    
    # =========================================================================
    # EQUITY CURVE
    # =========================================================================
    
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve",
        show_drawdown: bool = True,
        show_metrics: bool = True,
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot equity curve with optional benchmark and drawdown.
        
        Args:
            equity_curve: Portfolio value over time (DatetimeIndex)
            benchmark: Optional benchmark equity curve
            title: Plot title
            show_drawdown: Show drawdown subplot
            show_metrics: Show metrics annotation
            save_path: Path to save figure (None = don't save)
            show: Whether to display the plot
            close_after_save: Close figure after saving to free memory
            
        Returns:
            matplotlib Figure object
        """
        try:
            equity_curve = validate_series(equity_curve, "equity_curve")
        except ValueError as e:
            logger.error(f"Equity curve validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Create figure
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(self.figsize[0], self.figsize[1] + 3),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
        else:
            fig, ax1 = plt.subplots(figsize=self.figsize)
            ax2 = None
        
        # ----- Main Equity Curve -----
        ax1.plot(
            equity_curve.index,
            equity_curve.values,
            color=self.colors['primary'],
            linewidth=2.5,
            label='Strategy',
            zorder=3
        )
        
        # Fill under curve
        ax1.fill_between(
            equity_curve.index,
            equity_curve.iloc[0],
            equity_curve.values,
            color=self.colors['primary'],
            alpha=self.colors['fill_alpha'],
            zorder=2
        )
        
        # Benchmark
        if benchmark is not None and len(benchmark) > 0:
            try:
                benchmark = validate_series(benchmark, "benchmark")
                # Normalize to same starting point
                norm_benchmark = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
                ax1.plot(
                    norm_benchmark.index,
                    norm_benchmark.values,
                    color=self.colors['benchmark'],
                    linewidth=2,
                    linestyle='--',
                    label='Benchmark',
                    alpha=0.8,
                    zorder=2
                )
            except ValueError as e:
                logger.warning(f"Benchmark validation failed: {e}")
        
        # Formatting
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel(f'Portfolio Value ({self.currency_symbol})', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(self._format_currency))
        
        # Metrics annotation
        if show_metrics:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
            cagr = self._calculate_cagr(equity_curve)
            max_dd = self._calculate_max_drawdown(equity_curve) * 100
            
            metrics_text = (
                f"Total Return: {total_return:.1f}%\n"
                f"CAGR: {cagr:.1f}%\n"
                f"Max Drawdown: {max_dd:.1f}%"
            )
            
            ax1.annotate(
                metrics_text,
                xy=(0.02, 0.97),
                xycoords='axes fraction',
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor=self.colors['primary'],
                    alpha=0.9
                )
            )
        
        # ----- Drawdown Subplot -----
        if show_drawdown and ax2 is not None:
            drawdown = self._calculate_drawdown_series(equity_curve)
            
            ax2.fill_between(
                drawdown.index,
                drawdown.values * 100,
                0,
                color=self.colors['negative'],
                alpha=0.5
            )
            ax2.plot(
                drawdown.index,
                drawdown.values * 100,
                color=self.colors['negative'],
                linewidth=1
            )
            
            # Mark max drawdown point
            max_dd_idx = drawdown.idxmin()
            max_dd_val = drawdown.min() * 100
            
            ax2.scatter(
                [max_dd_idx], [max_dd_val],
                color='white',
                s=80,
                zorder=5,
                edgecolor=self.colors['negative'],
                linewidth=2
            )
            ax2.annotate(
                f'Max: {max_dd_val:.1f}%',
                xy=(max_dd_idx, max_dd_val),
                xytext=(15, 10),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=self.colors['negative'])
            )
            
            ax2.set_ylabel('Drawdown (%)', fontsize=11)
            ax2.set_xlabel('Date', fontsize=11)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_ylim(min(max_dd_val * 1.2, -5), 2)
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            ax1.set_xlabel('Date', fontsize=11)
        
        # Format x-axis
        self._format_date_axis(ax1)
        if ax2 is not None:
            self._format_date_axis(ax2)
        
        plt.tight_layout()
        
        # Save
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # ROLLING METRICS
    # =========================================================================
    
    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        windows: List[int] = [21, 63, 252],
        title: str = "Rolling Performance Metrics",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Daily returns series
            windows: List of rolling windows (trading days)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            close_after_save: Close figure after saving
            
        Returns:
            matplotlib Figure object
        """
        try:
            returns_clean = validate_series(returns, "returns", min_length=min(windows))
        except ValueError as e:
            logger.error(f"Returns validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], 14), sharex=True)
        
        window_labels = {21: '1M', 63: '3M', 126: '6M', 252: '1Y'}
        colors_list = ['#2E86AB', '#A23B72', '#28A745', '#FFC107']
        
        # Filter windows to available data
        valid_windows = [w for w in windows if len(returns_clean) >= w]
        
        if not valid_windows:
            logger.warning("No valid windows for data length")
            for ax in axes:
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                       transform=ax.transAxes)
            return fig
        
        # ----- 1. Rolling Returns -----
        ax1 = axes[0]
        for i, window in enumerate(valid_windows):
            rolling_ret = returns_clean.rolling(window).mean() * 252 * 100
            label = window_labels.get(window, f'{window}D')
            ax1.plot(rolling_ret.index, rolling_ret.values,
                    color=colors_list[i % len(colors_list)],
                    linewidth=1.5, label=label, alpha=0.85)
        
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax1.set_ylabel('Ann. Return (%)', fontsize=10)
        ax1.set_title('Rolling Annualized Returns', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=len(valid_windows))
        ax1.grid(True, alpha=0.3)
        
        # ----- 2. Rolling Volatility -----
        ax2 = axes[1]
        for i, window in enumerate(valid_windows):
            rolling_vol = returns_clean.rolling(window).std() * np.sqrt(252) * 100
            label = window_labels.get(window, f'{window}D')
            ax2.plot(rolling_vol.index, rolling_vol.values,
                    color=colors_list[i % len(colors_list)],
                    linewidth=1.5, label=label, alpha=0.85)
        
        ax2.set_ylabel('Ann. Volatility (%)', fontsize=10)
        ax2.set_title('Rolling Annualized Volatility', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, ncol=len(valid_windows))
        ax2.grid(True, alpha=0.3)
        
        # ----- 3. Rolling Sharpe Ratio -----
        ax3 = axes[2]
        for i, window in enumerate(valid_windows):
            rolling_mean = returns_clean.rolling(window).mean() * 252
            rolling_std = returns_clean.rolling(window).std() * np.sqrt(252)
            
            # Safe division to avoid division by zero
            rolling_sharpe = safe_divide(rolling_mean, rolling_std, default=0.0)
            
            label = window_labels.get(window, f'{window}D')
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                    color=colors_list[i % len(colors_list)],
                    linewidth=1.5, label=label, alpha=0.85)
        
        ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax3.axhline(1, color=self.colors['positive'], linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.axhline(2, color=self.colors['positive'], linestyle=':', linewidth=1, alpha=0.5)
        ax3.set_ylabel('Sharpe Ratio', fontsize=10)
        ax3.set_title('Rolling Sharpe Ratio', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9, ncol=len(valid_windows))
        ax3.grid(True, alpha=0.3)
        
        # ----- 4. Rolling Max Drawdown -----
        ax4 = axes[3]
        for i, window in enumerate(valid_windows):
            rolling_dd = returns_clean.rolling(window).apply(
                lambda x: self._max_dd_from_returns(x), raw=False
            ) * 100
            label = window_labels.get(window, f'{window}D')
            ax4.plot(rolling_dd.index, rolling_dd.values,
                    color=colors_list[i % len(colors_list)],
                    linewidth=1.5, label=label, alpha=0.85)
        
        ax4.set_ylabel('Max Drawdown (%)', fontsize=10)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_title('Rolling Maximum Drawdown', fontsize=11, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=9, ncol=len(valid_windows))
        ax4.grid(True, alpha=0.3)
        
        self._format_date_axis(ax4)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # RETURNS DISTRIBUTION
    # =========================================================================
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot returns distribution with statistics.
        
        Args:
            returns: Returns series
            title: Plot title
            save_path: Path to save
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            returns_clean = validate_series(returns, "returns")
        except ValueError as e:
            logger.error(f"Returns validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        returns_pct = returns_clean * 100  # Convert to %
        
        # ----- 1. Histogram with Normal Overlay -----
        ax1 = axes[0, 0]
        ax1.hist(returns_pct, bins=50, alpha=0.7, color=self.colors['primary'],
                edgecolor='white', density=True, label='Returns')
        
        # Normal distribution overlay
        mu, sigma = returns_pct.mean(), returns_pct.std()
        x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma),
                color=self.colors['negative'], linewidth=2, 
                linestyle='--', label='Normal Dist')
        
        ax1.set_title('Distribution vs Normal', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Return (%)', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # ----- 2. Q-Q Plot -----
        ax2 = axes[0, 1]
        stats.probplot(returns_pct, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ----- 3. Box Plot -----
        ax3 = axes[1, 0]
        bp = ax3.boxplot(returns_pct, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=self.colors['primary'], alpha=0.6),
                        medianprops=dict(color=self.colors['negative'], linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Return (%)', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticklabels(['Returns'])
        
        # ----- 4. Statistics Table -----
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate Sharpe (annualized)
        sharpe = safe_divide(returns_clean.mean(), returns_clean.std(), 0) * np.sqrt(252)
        
        stats_data = [
            ['Mean', f'{returns_pct.mean():.3f}%'],
            ['Median', f'{returns_pct.median():.3f}%'],
            ['Std Dev', f'{returns_pct.std():.3f}%'],
            ['Skewness', f'{stats.skew(returns_pct):.3f}'],
            ['Kurtosis', f'{stats.kurtosis(returns_pct):.3f}'],
            ['Min', f'{returns_pct.min():.3f}%'],
            ['Max', f'{returns_pct.max():.3f}%'],
            ['Sharpe (ann.)', f'{sharpe:.3f}']
        ]
        
        table = ax4.table(cellText=stats_data, cellLoc='left',
                         colWidths=[0.5, 0.5], loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(stats_data)):
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 0)].set_text_props(weight='bold')
            table[(i, 1)].set_facecolor('white')
        
        ax4.set_title('Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # MONTHLY HEATMAP
    # =========================================================================
    
    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap (%)",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            returns: Daily returns series
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            returns_clean = validate_series(returns, "returns")
        except ValueError as e:
            logger.error(f"Returns validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Resample to monthly
        monthly_returns = returns_clean.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        if len(monthly_returns) == 0:
            logger.warning("No monthly data available")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No Monthly Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return fig
        
        # Create pivot table
        monthly_df = pd.DataFrame({
            'return': monthly_returns.values * 100,
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
        
        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return (%)'},
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            vmin=-10,  # Reasonable bounds
            vmax=10
        )
        
        # Format
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Year', fontsize=11)
        
        # Month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # TRADE ANALYSIS
    # =========================================================================
    
    def plot_trade_analysis(
        self,
        trades_df: pd.DataFrame,
        pnl_col: str = 'pnl',
        return_col: str = 'return_pct',
        entry_col: str = 'entry_date',
        exit_col: str = 'exit_date',
        title: str = "Trade Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot trade analysis.
        
        Args:
            trades_df: DataFrame with trade data
            pnl_col: PnL column name
            return_col: Return % column name
            entry_col: Entry date column
            exit_col: Exit date column
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            validate_dataframe(trades_df, [pnl_col, return_col], "trades_df")
        except ValueError as e:
            logger.error(f"Trades validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ----- 1. Cumulative PnL -----
        ax1 = axes[0, 0]
        cumulative_pnl = trades_df[pnl_col].cumsum()
        
        if exit_col in trades_df.columns:
            x_vals = trades_df[exit_col]
        else:
            x_vals = range(len(trades_df))
        
        ax1.plot(x_vals, cumulative_pnl,
                color=self.colors['primary'], linewidth=2)
        ax1.fill_between(x_vals, 0, cumulative_pnl,
                         color=self.colors['primary'], alpha=0.2)
        ax1.set_title('Cumulative PnL', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'PnL ({self.currency_symbol})', fontsize=10)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(self._format_currency))
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linewidth=0.5)
        
        # ----- 2. Win/Loss Distribution -----
        ax2 = axes[0, 1]
        wins = trades_df[trades_df[pnl_col] > 0][pnl_col]
        losses = trades_df[trades_df[pnl_col] <= 0][pnl_col]
        
        if len(wins) > 0 and len(losses) > 0:
            ax2.hist([wins, losses], bins=30, label=['Wins', 'Losses'],
                    color=[self.colors['positive'], self.colors['negative']],
                    alpha=0.7, edgecolor='white')
        elif len(wins) > 0:
            ax2.hist(wins, bins=30, label='Wins',
                    color=self.colors['positive'], alpha=0.7)
        elif len(losses) > 0:
            ax2.hist(losses, bins=30, label='Losses',
                    color=self.colors['negative'], alpha=0.7)
        
        ax2.set_title('PnL Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'PnL ({self.currency_symbol})', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ----- 3. Trade Returns -----
        ax3 = axes[1, 0]
        colors = [self.colors['positive'] if r > 0 else self.colors['negative']
                 for r in trades_df[return_col]]
        ax3.bar(range(len(trades_df)), trades_df[return_col] * 100,
               color=colors, alpha=0.6)
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_title('Trade Returns', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade Number', fontsize=10)
        ax3.set_ylabel('Return (%)', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ----- 4. Statistics Table -----
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        win_rate = (trades_df[pnl_col] > 0).sum() / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
        
        stats_data = [
            ['Total Trades', f'{len(trades_df)}'],
            ['Win Rate', f'{win_rate:.1f}%'],
            ['Avg Win', self._format_currency(avg_win, 0)],
            ['Avg Loss', self._format_currency(avg_loss, 0)],
            ['Profit Factor', f'{profit_factor:.2f}' if profit_factor != np.inf else '∞'],
            ['Total PnL', self._format_currency(trades_df[pnl_col].sum(), 0)],
            ['Best Trade', self._format_currency(trades_df[pnl_col].max(), 0)],
            ['Worst Trade', self._format_currency(trades_df[pnl_col].min(), 0)]
        ]
        
        table = ax4.table(cellText=stats_data, cellLoc='left',
                         colWidths=[0.5, 0.5], loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(stats_data)):
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 0)].set_text_props(weight='bold')
            table[(i, 1)].set_facecolor('white')
        
        ax4.set_title('Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig


# =============================================================================
# FACTOR PLOTTER
# =============================================================================

class FactorPlotter:
    """Visualization for factor/feature analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 150):
        """Initialize factor plotter."""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'positive': '#28A745',
            'negative': '#DC3545',
            'warning': '#FFC107'
        }
    
    def _save_figure(self, fig: plt.Figure, path: str) -> None:
        """Save figure to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        print(f"✅ Saved: {path}")
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        feature_col: str = 'feature',
        importance_col: str = 'importance',
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with features and importance
            top_n: Number of top features to show
            feature_col: Feature name column
            importance_col: Importance value column
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            validate_dataframe(importance_df, [feature_col, importance_col], "importance_df")
        except ValueError as e:
            logger.error(f"Importance validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Get top N
        df_sorted = importance_df.nlargest(top_n, importance_col)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(6, top_n * 0.3)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(df_sorted))
        bars = ax.barh(y_pos, df_sorted[importance_col].values,
                      color=self.colors['primary'], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted[feature_col].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, v) in enumerate(zip(bars, df_sorted[importance_col].values)):
            ax.text(v, i, f' {v:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_ic_decay(
        self,
        decay_df: pd.DataFrame,
        horizon_col: str = 'horizon',
        ic_col: str = 'ic',
        rank_ic_col: Optional[str] = 'rank_ic',
        title: str = "Alpha Decay Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot IC decay over horizons.
        
        Args:
            decay_df: DataFrame with horizons and IC values
            horizon_col: Horizon column name
            ic_col: IC column name
            rank_ic_col: Rank IC column name (optional)
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            validate_dataframe(decay_df, [horizon_col, ic_col], "decay_df")
        except ValueError as e:
            logger.error(f"Decay validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # IC line
        ax.plot(decay_df[horizon_col], decay_df[ic_col],
               marker='o', linewidth=2.5, markersize=8,
               color=self.colors['primary'], label='IC')
        
        # Rank IC line (if provided)
        if rank_ic_col and rank_ic_col in decay_df.columns:
            ax.plot(decay_df[horizon_col], decay_df[rank_ic_col],
                   marker='s', linewidth=2.5, markersize=8,
                   color=self.colors['secondary'], label='Rank IC',
                   linestyle='--')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Prediction Horizon (days)', fontsize=11)
        ax.set_ylabel('Information Coefficient', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Feature Correlation Matrix",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        if corr_matrix is None or len(corr_matrix) == 0:
            logger.error("Correlation matrix is empty")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'Empty Matrix', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        n_features = len(corr_matrix)
        fig, ax = plt.subplots(figsize=(max(10, n_features * 0.5),
                                       max(8, n_features * 0.4)))
        
        sns.heatmap(
            corr_matrix,
            annot=True if n_features <= 15 else False,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_ic_series(
        self,
        ic_series: pd.Series,
        title: str = "Information Coefficient Over Time",
        rolling_window: int = 6,
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot IC time series with rolling average.
        
        Args:
            ic_series: IC values over time
            title: Plot title
            rolling_window: Window for rolling mean
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            ic_series = validate_series(ic_series, "ic_series")
        except ValueError as e:
            logger.error(f"IC series validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # IC bars
        colors = [self.colors['positive'] if x > 0 else self.colors['negative']
                 for x in ic_series.values]
        ax.bar(ic_series.index, ic_series.values,
              color=colors, alpha=0.6, width=20, label='IC')
        
        # Rolling average
        if len(ic_series) >= rolling_window:
            rolling_ic = ic_series.rolling(rolling_window).mean()
            ax.plot(rolling_ic.index, rolling_ic.values,
                   color=self.colors['primary'], linewidth=2.5,
                   label=f'{rolling_window}-period MA')
        
        # Mean line
        mean_ic = ic_series.mean()
        ax.axhline(mean_ic, color=self.colors['primary'], linestyle=':',
                  linewidth=1.5, label=f'Mean: {mean_ic:.4f}')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Information Coefficient', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig


# =============================================================================
# RISK PLOTTER
# =============================================================================

class RiskPlotter:
    """Risk metrics visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 150):
        """Initialize risk plotter."""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'positive': '#28A745',
            'negative': '#DC3545',
            'warning': '#FFC107'
        }
    
    def _save_figure(self, fig: plt.Figure, path: str) -> None:
        """Save figure to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        print(f"✅ Saved: {path}")
    
    def plot_var_analysis(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        title: str = "Value at Risk Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot VaR analysis.
        
        Args:
            returns: Returns series
            confidence_levels: VaR confidence levels
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            returns_clean = validate_series(returns, "returns")
        except ValueError as e:
            logger.error(f"Returns validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        returns_pct = returns_clean * 100
        
        # Histogram
        ax.hist(returns_pct, bins=50, alpha=0.7, color=self.colors['primary'],
               density=True, edgecolor='white')
        
        # VaR lines
        colors = [self.colors['warning'], self.colors['negative']]
        for i, conf in enumerate(confidence_levels):
            var = np.percentile(returns_pct, (1 - conf) * 100)
            ax.axvline(var, color=colors[i], linewidth=2.5, linestyle='--',
                      label=f'VaR {conf*100:.0f}%: {var:.2f}%')
        
        ax.set_xlabel('Return (%)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 21,
        title: str = "Rolling Volatility",
        save_path: Optional[str] = None,
        show: bool = True,
        close_after_save: bool = True
    ) -> plt.Figure:
        """
        Plot rolling volatility.
        
        Args:
            returns: Returns series
            window: Rolling window
            title: Plot title
            save_path: Save path
            show: Display plot
            close_after_save: Close after saving
            
        Returns:
            Figure object
        """
        try:
            returns_clean = validate_series(returns, "returns", min_length=window)
        except ValueError as e:
            logger.error(f"Returns validation failed: {e}")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate rolling volatility (annualized)
        rolling_vol = returns_clean.rolling(window).std() * np.sqrt(252) * 100
        
        ax.plot(rolling_vol.index, rolling_vol.values,
               color=self.colors['primary'], linewidth=2)
        ax.fill_between(rolling_vol.index, 0, rolling_vol.values,
                       color=self.colors['primary'], alpha=0.2)
        
        # Mean line
        mean_vol = rolling_vol.mean()
        ax.axhline(mean_vol, color=self.colors['negative'], linestyle='--',
                  linewidth=1.5, label=f'Mean: {mean_vol:.2f}%')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
            if close_after_save and not show:
                plt.close(fig)
        
        if show:
            plt.show()
        
        return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Quick function to plot equity curve."""
    plotter = PerformancePlotter(**kwargs)
    return plotter.plot_equity_curve(
        equity_curve, benchmark, save_path=save_path, show=show
    )


def plot_drawdown(
    equity_curve: pd.Series,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Quick function to plot drawdown."""
    plotter = PerformancePlotter(**kwargs)
    
    fig, ax = plt.subplots(figsize=plotter.figsize)
    
    drawdown = plotter._calculate_drawdown_series(equity_curve) * 100
    
    ax.fill_between(drawdown.index, drawdown.values, 0,
                   color=plotter.colors['negative'], alpha=0.5)
    ax.plot(drawdown.index, drawdown.values,
           color=plotter.colors['negative'], linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    
    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plotter._save_figure(fig, save_path)
        if not show:
            plt.close(fig)
    
    if show:
        plt.show()
    
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Quick function to plot returns distribution."""
    plotter = PerformancePlotter(**kwargs)
    return plotter.plot_returns_distribution(returns, save_path=save_path, show=show)


def plot_ic_series(
    ic_series: pd.Series,
    title: str = "Information Coefficient Over Time",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Quick function to plot IC series."""
    plotter = FactorPlotter(**kwargs)
    return plotter.plot_ic_series(ic_series, title, save_path=save_path, show=show)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Quick function to plot feature importance."""
    plotter = FactorPlotter(**kwargs)
    return plotter.plot_feature_importance(
        importance_df, top_n, save_path=save_path, show=show
    )


def plot_quantile_returns(
    predictions_df: pd.DataFrame,
    n_quantiles: int = 5,
    pred_col: str = 'prediction',
    return_col: str = 'forward_return',
    date_col: str = 'date',
    title: str = "Returns by Prediction Quantile",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot average returns by prediction quantile.
    
    Args:
        predictions_df: DataFrame with predictions and returns
        n_quantiles: Number of quantiles
        pred_col: Prediction column name
        return_col: Return column name
        date_col: Date column name
        title: Plot title
        save_path: Save path
        show: Display plot
        
    Returns:
        Figure object
    """
    try:
        validate_dataframe(predictions_df, [pred_col, return_col, date_col], 
                          "predictions_df")
    except ValueError as e:
        logger.error(f"Predictions validation failed: {e}")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, str(e), ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='red')
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df = predictions_df.copy()
    
    try:
        # Calculate quantiles per date with error handling
        def assign_quantile(x):
            try:
                return pd.qcut(x, n_quantiles, labels=range(1, n_quantiles + 1),
                              duplicates='drop')
            except ValueError:
                # If qcut fails, use rank-based assignment
                return pd.cut(x.rank(method='first'), n_quantiles,
                            labels=range(1, n_quantiles + 1))
        
        df['quantile'] = df.groupby(date_col)[pred_col].transform(assign_quantile)
        
        # Remove NaN quantiles
        df = df.dropna(subset=['quantile'])
        
        if len(df) == 0:
            logger.warning("No valid quantiles generated")
            for ax in axes:
                ax.text(0.5, 0.5, 'Insufficient Data',
                       ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Average return by quantile
        quantile_returns = df.groupby('quantile')[return_col].mean() * 100
        
        # ----- 1. Bar Chart -----
        ax1 = axes[0]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quantile_returns)))
        
        bars = ax1.bar(quantile_returns.index.astype(str), quantile_returns.values,
                      color=colors, alpha=0.8, edgecolor='white')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax1.set_xlabel('Prediction Quantile', fontsize=11)
        ax1.set_ylabel('Avg Return (%)', fontsize=11)
        ax1.set_title('Avg Return by Quantile', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.3f}%', ha='center',
                    va='bottom' if height > 0 else 'top', fontsize=9)
        
        # ----- 2. Long-Short Spread -----
        ax2 = axes[1]
        
        spread_data = df.groupby(date_col).apply(
            lambda x: (x[x['quantile'] == n_quantiles][return_col].mean() -
                      x[x['quantile'] == 1][return_col].mean())
        ).dropna() * 100
        
        if len(spread_data) > 0:
            colors_spread = ['#28A745' if x > 0 else '#DC3545'
                           for x in spread_data.values]
            ax2.bar(range(len(spread_data)), spread_data.values,
                   color=colors_spread, alpha=0.6)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax2.axhline(spread_data.mean(), color='#2E86AB', linestyle=':',
                       linewidth=2, label=f'Mean: {spread_data.mean():.3f}%')
            ax2.set_xlabel('Period', fontsize=11)
            ax2.set_ylabel('Q5 - Q1 Return (%)', fontsize=11)
            ax2.set_title('Long-Short Spread', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
    except Exception as e:
        logger.error(f"Error in plot_quantile_returns: {e}")
        for ax in axes:
            ax.text(0.5, 0.5, f'Error: {str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        if not show:
            plt.close(fig)
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# COMPREHENSIVE REPORT
# =============================================================================

def quick_plot_all(
    equity_curve: pd.Series,
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    trades_df: Optional[pd.DataFrame] = None,
    ic_series: Optional[pd.Series] = None,
    importance_df: Optional[pd.DataFrame] = None,
    output_dir: str = "output/plots",
    show: bool = False,
    **kwargs
) -> Dict[str, plt.Figure]:
    """
    Generate all standard plots and save to directory.
    
    Args:
        equity_curve: Portfolio equity curve
        returns: Returns series
        benchmark: Benchmark equity curve (optional)
        trades_df: Trade data (optional)
        ic_series: IC time series (optional)
        importance_df: Feature importance (optional)
        output_dir: Output directory
        show: Display plots
        **kwargs: Additional plotter arguments
        
    Returns:
        Dictionary of figure objects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating all plots to: {output_dir}")
    print(f"\n📊 Generating Performance Report...")
    print(f"📁 Output: {output_dir}")
    
    figures = {}
    perf_plotter = PerformancePlotter(**kwargs)
    factor_plotter = FactorPlotter()
    risk_plotter = RiskPlotter()
    
    # 1. Equity Curve
    print("  ├─ Equity curve...")
    figures['equity'] = perf_plotter.plot_equity_curve(
        equity_curve, benchmark,
        save_path=output_path / "equity_curve.png",
        show=show,
        close_after_save=True
    )
    
    # 2. Rolling Metrics
    print("  ├─ Rolling metrics...")
    figures['rolling'] = perf_plotter.plot_rolling_metrics(
        returns,
        save_path=output_path / "rolling_metrics.png",
        show=show,
        close_after_save=True
    )
    
    # 3. Returns Distribution
    print("  ├─ Returns distribution...")
    figures['returns_dist'] = perf_plotter.plot_returns_distribution(
        returns,
        save_path=output_path / "returns_distribution.png",
        show=show,
        close_after_save=True
    )
    
    # 4. Monthly Heatmap
    print("  ├─ Monthly heatmap...")
    figures['monthly'] = perf_plotter.plot_monthly_heatmap(
        returns,
        save_path=output_path / "monthly_heatmap.png",
        show=show,
        close_after_save=True
    )
    
    # 5. Trade Analysis (if provided)
    if trades_df is not None and len(trades_df) > 0:
        print("  ├─ Trade analysis...")
        figures['trades'] = perf_plotter.plot_trade_analysis(
            trades_df,
            save_path=output_path / "trade_analysis.png",
            show=show,
            close_after_save=True
        )
    
    # 6. IC Series (if provided)
    if ic_series is not None and len(ic_series) > 0:
        print("  ├─ IC series...")
        figures['ic'] = factor_plotter.plot_ic_series(
            ic_series,
            save_path=output_path / "ic_series.png",
            show=show,
            close_after_save=True
        )
    
    # 7. Feature Importance (if provided)
    if importance_df is not None and len(importance_df) > 0:
        print("  ├─ Feature importance...")
        figures['importance'] = factor_plotter.plot_feature_importance(
            importance_df,
            save_path=output_path / "feature_importance.png",
            show=show,
            close_after_save=True
        )
    
    # 8. VaR Analysis
    print("  ├─ VaR analysis...")
    figures['var'] = risk_plotter.plot_var_analysis(
        returns,
        save_path=output_path / "var_analysis.png",
        show=show,
        close_after_save=True
    )
    
    # 9. Rolling Volatility
    print("  └─ Rolling volatility...")
    figures['vol'] = risk_plotter.plot_rolling_volatility(
        returns,
        save_path=output_path / "rolling_volatility.png",
        show=show,
        close_after_save=True
    )
    
    print(f"\n✅ Generated {len(figures)} plots")
    print(f"📂 Saved to: {output_dir}")
    
    return figures


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate sample data
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    equity = (1 + returns).cumprod() * 100000
    
    # Test plots
    print("Testing plots.py...")
    
    plotter = PerformancePlotter(currency_symbol='$', indian_format=False)
    
    # Test equity curve
    fig = plotter.plot_equity_curve(equity, show=False)
    print("✅ Equity curve generated")
    plt.close(fig)
    
    # Test returns distribution
    fig = plotter.plot_returns_distribution(returns, show=False)
    print("✅ Returns distribution generated")
    plt.close(fig)
    
    # Test monthly heatmap
    fig = plotter.plot_monthly_heatmap(returns, show=False)
    print("✅ Monthly heatmap generated")
    plt.close(fig)
    
    print("\n✅ All tests passed!")