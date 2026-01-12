"""
Performance Charts & Visualization - Pure Python
=================================================
Professional static plots using matplotlib and seaborn.
No HTML - only Python!

Classes:
    - PerformancePlotter: Equity curves, returns, drawdowns
    - FactorPlotter: Feature importance, factor analysis
    - RiskPlotter: Risk metrics, VaR, stress testing

Output Formats:
    - PNG, JPG, SVG, PDF (single charts)
    - Display in Jupyter notebooks
    - Save to file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


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
        
        # Or save directly
        >>> plotter.plot_equity_curve(equity_curve, save_path='equity.png')
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        style: str = 'default',
        dpi: int = 150
    ):
        """
        Initialize plotter.
        
        Args:
            figsize: Default figure size (width, height)
            style: Plot style ('default', 'dark', 'minimal')
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self._setup_colors()
    
    def _setup_colors(self):
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
                'grid_color': '#444444'
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
                'grid_color': '#E0E0E0'
            }
    
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
        show: bool = True
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
            
        Returns:
            matplotlib Figure object
        """
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
        if benchmark is not None:
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
        
        # Formatting
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
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
        if ax2:
            self._format_date_axis(ax2)
        
        plt.tight_layout()
        
        # Save
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # RETURNS DISTRIBUTION
    # =========================================================================
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Returns Distribution Analysis",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot comprehensive returns distribution analysis.
        
        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] + 4))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Clean data
        returns_clean = returns.dropna()
        
        # ----- 1. Histogram with KDE -----
        ax1 = fig.add_subplot(gs[0, :2])
        
        returns_pct = returns_clean * 100
        
        # Histogram
        n, bins, patches = ax1.hist(
            returns_pct,
            bins=50,
            density=True,
            alpha=0.7,
            color=self.colors['primary'],
            edgecolor='white',
            linewidth=0.5
        )
        
        # Color negative bins red
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor(self.colors['negative'])
                patch.set_alpha(0.6)
        
        # KDE
        if len(returns_pct) > 1:
            kde_x = np.linspace(returns_pct.min(), returns_pct.max(), 200)
            kde = stats.gaussian_kde(returns_pct)
            ax1.plot(kde_x, kde(kde_x), color=self.colors['primary'], linewidth=2.5, label='KDE')
            
            # Normal distribution overlay
            mu, sigma = returns_pct.mean(), returns_pct.std()
            normal_y = stats.norm.pdf(kde_x, mu, sigma)
            ax1.plot(kde_x, normal_y, color=self.colors['benchmark'], 
                    linewidth=2, linestyle='--', label='Normal', alpha=0.8)
        
        # Mean line
        mu = returns_pct.mean()
        ax1.axvline(mu, color=self.colors['positive'], linewidth=2, 
                   linestyle='-', label=f'Mean: {mu:.3f}%')
        ax1.axvline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Daily Return (%)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Return Distribution with KDE', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ----- 2. Statistics Box -----
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats_dict = {
            'Mean': f'{returns_clean.mean()*100:.4f}%',
            'Median': f'{returns_clean.median()*100:.4f}%',
            'Std Dev': f'{returns_clean.std()*100:.4f}%',
            'Skewness': f'{returns_clean.skew():.4f}',
            'Kurtosis': f'{returns_clean.kurtosis():.4f}',
            'Min': f'{returns_clean.min()*100:.2f}%',
            'Max': f'{returns_clean.max()*100:.2f}%',
            'VaR (5%)': f'{np.percentile(returns_clean, 5)*100:.2f}%',
            'Positive Days': f'{(returns_clean > 0).mean()*100:.1f}%'
        }
        
        y_pos = 0.95
        ax2.text(0.1, y_pos, 'Distribution Statistics', fontsize=12, 
                fontweight='bold', transform=ax2.transAxes)
        y_pos -= 0.08
        
        for key, value in stats_dict.items():
            ax2.text(0.1, y_pos, f'{key}:', fontsize=10, 
                    transform=ax2.transAxes, fontfamily='monospace')
            ax2.text(0.7, y_pos, value, fontsize=10, 
                    transform=ax2.transAxes, fontfamily='monospace',
                    ha='right')
            y_pos -= 0.08
        
        # ----- 3. Q-Q Plot -----
        ax3 = fig.add_subplot(gs[1, 0])
        
        if len(returns_clean) > 1:
            stats.probplot(returns_clean, dist="norm", plot=ax3)
            ax3.get_lines()[0].set_markerfacecolor(self.colors['primary'])
            ax3.get_lines()[0].set_markeredgecolor(self.colors['primary'])
            ax3.get_lines()[0].set_markersize(4)
            ax3.get_lines()[1].set_color(self.colors['negative'])
        ax3.set_title('Q-Q Plot (Normality)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ----- 4. Box Plot by Month -----
        ax4 = fig.add_subplot(gs[1, 1])
        
        monthly_data = []
        month_labels = []
        for month in range(1, 13):
            month_returns = returns_clean[returns_clean.index.month == month] * 100
            if len(month_returns) > 0:
                monthly_data.append(month_returns.values)
                month_labels.append(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1])
        
        if monthly_data:
            bp = ax4.boxplot(monthly_data, labels=month_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(self.colors['primary'])
                patch.set_alpha(0.6)
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(2)
        
        ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax4.set_xlabel('Month', fontsize=10)
        ax4.set_ylabel('Daily Return (%)', fontsize=10)
        ax4.set_title('Returns by Month', fontsize=11, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ----- 5. Cumulative Returns -----
        ax5 = fig.add_subplot(gs[1, 2])
        
        cumulative = (1 + returns_clean).cumprod() - 1
        
        ax5.plot(cumulative.index, cumulative.values * 100,
                color=self.colors['primary'], linewidth=2)
        
        # Fill positive and negative areas
        ax5.fill_between(
            cumulative.index, cumulative.values * 100, 0,
            where=(cumulative.values >= 0),
            color=self.colors['positive'], alpha=0.3
        )
        ax5.fill_between(
            cumulative.index, cumulative.values * 100, 0,
            where=(cumulative.values < 0),
            color=self.colors['negative'], alpha=0.3
        )
        
        ax5.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax5.set_title('Cumulative Returns', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        self._format_date_axis(ax5)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        
        if save_path:
            self._save_figure(fig, save_path)
        
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
        show: bool = True
    ) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Daily returns series
            windows: List of rolling windows (trading days)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], 14), sharex=True)
        
        window_labels = {21: '1M', 63: '3M', 126: '6M', 252: '1Y'}
        colors_list = ['#2E86AB', '#A23B72', '#28A745', '#FFC107']
        
        returns_clean = returns.dropna()
        
        # ----- 1. Rolling Returns -----
        ax1 = axes[0]
        for i, window in enumerate(windows):
            if len(returns_clean) >= window:
                rolling_ret = returns_clean.rolling(window).mean() * 252 * 100  # Annualized
                label = window_labels.get(window, f'{window}D')
                ax1.plot(rolling_ret.index, rolling_ret.values,
                        color=colors_list[i % len(colors_list)],
                        linewidth=1.5, label=label, alpha=0.85)
        
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax1.set_ylabel('Ann. Return (%)', fontsize=10)
        ax1.set_title('Rolling Annualized Returns', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=len(windows))
        ax1.grid(True, alpha=0.3)
        
        # ----- 2. Rolling Volatility -----
        ax2 = axes[1]
        for i, window in enumerate(windows):
            if len(returns_clean) >= window:
                rolling_vol = returns_clean.rolling(window).std() * np.sqrt(252) * 100
                label = window_labels.get(window, f'{window}D')
                ax2.plot(rolling_vol.index, rolling_vol.values,
                        color=colors_list[i % len(colors_list)],
                        linewidth=1.5, label=label, alpha=0.85)
        
        ax2.set_ylabel('Ann. Volatility (%)', fontsize=10)
        ax2.set_title('Rolling Annualized Volatility', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, ncol=len(windows))
        ax2.grid(True, alpha=0.3)
        
        # ----- 3. Rolling Sharpe Ratio -----
        ax3 = axes[2]
        for i, window in enumerate(windows):
            if len(returns_clean) >= window:
                rolling_mean = returns_clean.rolling(window).mean() * 252
                rolling_std = returns_clean.rolling(window).std() * np.sqrt(252)
                rolling_sharpe = rolling_mean / rolling_std
                
                label = window_labels.get(window, f'{window}D')
                ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                        color=colors_list[i % len(colors_list)],
                        linewidth=1.5, label=label, alpha=0.85)
        
        ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax3.axhline(1, color=self.colors['positive'], linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.axhline(2, color=self.colors['positive'], linestyle=':', linewidth=1, alpha=0.5)
        ax3.set_ylabel('Sharpe Ratio', fontsize=10)
        ax3.set_title('Rolling Sharpe Ratio', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9, ncol=len(windows))
        ax3.grid(True, alpha=0.3)
        
        # ----- 4. Rolling Max Drawdown -----
        ax4 = axes[3]
        for i, window in enumerate(windows):
            if len(returns_clean) >= window:
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
        ax4.legend(loc='lower right', fontsize=9, ncol=len(windows))
        ax4.grid(True, alpha=0.3)
        
        self._format_date_axis(ax4)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
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
        show: bool = True
    ) -> plt.Figure:
        """
        Plot monthly returns as a heatmap.
        
        Args:
            returns: Daily returns series
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        returns_clean = returns.dropna()
        
        # Calculate monthly returns
        monthly_returns = returns_clean.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })
        
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Add yearly total
        pivot['Year Total'] = pivot.sum(axis=1)
        
        # Create figure
        fig_height = max(6, len(pivot) * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        
        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Return (%)', 'shrink': 0.8},
            annot_kws={'size': 9, 'fontweight': 'bold'},
            linewidths=1,
            linecolor='white'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Year', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # TRADE ANALYSIS (FIXED VERSION)
    # =========================================================================
    
    def plot_trade_analysis(
        self,
        trades: pd.DataFrame,
        title: str = "Trade Analysis",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Comprehensive trade analysis visualization.
        
        Args:
            trades: DataFrame with columns: date, ticker, side, pnl, return_pct
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Make a copy to avoid modifying original
        trades = trades.copy()
        
        # ----- 1. Win/Loss Distribution -----
        ax1 = fig.add_subplot(gs[0, 0])
        
        if 'pnl' in trades.columns:
            wins = trades[trades['pnl'] > 0]['pnl'] / 1000  # In thousands
            losses = trades[trades['pnl'] <= 0]['pnl'] / 1000
            
            if len(wins) > 0:
                ax1.hist(wins, bins=min(30, max(10, len(wins))), color=self.colors['positive'],
                        alpha=0.7, label=f'Wins ({len(wins)})', edgecolor='white')
            if len(losses) > 0:
                ax1.hist(losses, bins=min(30, max(10, len(losses))), color=self.colors['negative'],
                        alpha=0.7, label=f'Losses ({len(losses)})', edgecolor='white')
            ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
            
            ax1.set_xlabel('P&L (₹ Thousands)', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.set_title('Win/Loss Distribution', fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
        
        # ----- 2. Win Rate by Month -----
        ax2 = fig.add_subplot(gs[0, 1])
        
        if 'date' in trades.columns and 'pnl' in trades.columns:
            trades['date'] = pd.to_datetime(trades['date'])
            trades['month'] = trades['date'].dt.to_period('M')
            
            monthly_wr = trades.groupby('month').apply(
                lambda x: (x['pnl'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
            )
            
            if len(monthly_wr) > 0:
                colors = [self.colors['positive'] if x >= 50 else self.colors['negative'] 
                         for x in monthly_wr.values]
                
                x_positions = range(len(monthly_wr))
                ax2.bar(x_positions, monthly_wr.values, color=colors, alpha=0.8, edgecolor='white')
                ax2.axhline(50, color='gray', linestyle='--', linewidth=1.5)
                
                # X-axis labels (show every nth month)
                n_labels = min(12, len(monthly_wr))
                step = max(1, len(monthly_wr) // n_labels)
                ax2.set_xticks(range(0, len(monthly_wr), step))
                ax2.set_xticklabels(
                    [str(monthly_wr.index[i])[-5:] for i in range(0, len(monthly_wr), step)],
                    rotation=45, ha='right', fontsize=8
                )
            
            ax2.set_xlabel('Month', fontsize=10)
            ax2.set_ylabel('Win Rate (%)', fontsize=10)
            ax2.set_title('Monthly Win Rate', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # ----- 3. P&L by Day of Week (FIXED) -----
        ax3 = fig.add_subplot(gs[0, 2])
        
        if 'date' in trades.columns and 'pnl' in trades.columns:
            trades['date'] = pd.to_datetime(trades['date'])
            trades['dow'] = trades['date'].dt.dayofweek
            dow_pnl = trades.groupby('dow')['pnl'].mean() / 1000
            
            # All days mapping
            all_days = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            
            # Create aligned data
            plot_days = []
            plot_values = []
            plot_colors = []
            
            for day_num in sorted(dow_pnl.index):
                if day_num in all_days:
                    plot_days.append(all_days[day_num])
                    plot_values.append(dow_pnl[day_num])
                    color = self.colors['positive'] if dow_pnl[day_num] > 0 else self.colors['negative']
                    plot_colors.append(color)
            
            if plot_days:
                ax3.bar(plot_days, plot_values, color=plot_colors, alpha=0.8, edgecolor='white')
                ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            ax3.set_xlabel('Day of Week', fontsize=10)
            ax3.set_ylabel('Avg P&L (₹K)', fontsize=10)
            ax3.set_title('Avg P&L by Weekday', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # ----- 4. Cumulative P&L -----
        ax4 = fig.add_subplot(gs[1, 0])
        
        if 'pnl' in trades.columns:
            cum_pnl = trades['pnl'].cumsum() / 100000  # In Lakhs
            
            ax4.plot(range(len(cum_pnl)), cum_pnl.values,
                    color=self.colors['primary'], linewidth=2)
            ax4.fill_between(
                range(len(cum_pnl)), cum_pnl.values, 0,
                where=(cum_pnl.values >= 0),
                color=self.colors['positive'], alpha=0.3
            )
            ax4.fill_between(
                range(len(cum_pnl)), cum_pnl.values, 0,
                where=(cum_pnl.values < 0),
                color=self.colors['negative'], alpha=0.3
            )
            ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            ax4.set_xlabel('Trade Number', fontsize=10)
            ax4.set_ylabel('Cumulative P&L (₹ Lakhs)', fontsize=10)
            ax4.set_title('Cumulative P&L', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # ----- 5. Top/Bottom Performers -----
        ax5 = fig.add_subplot(gs[1, 1])
        
        if 'ticker' in trades.columns and 'pnl' in trades.columns:
            ticker_pnl = trades.groupby('ticker')['pnl'].sum().sort_values() / 1000
            
            # Top 5 and Bottom 5
            n_show = min(5, max(1, len(ticker_pnl) // 2))
            
            if n_show > 0 and len(ticker_pnl) >= 2:
                bottom_n = ticker_pnl.head(n_show)
                top_n = ticker_pnl.tail(n_show)
                combined = pd.concat([bottom_n, top_n])
                
                colors = [self.colors['negative'] if x < 0 else self.colors['positive']
                         for x in combined.values]
                
                y_positions = range(len(combined))
                ax5.barh(y_positions, combined.values, color=colors, alpha=0.8, edgecolor='white')
                ax5.set_yticks(y_positions)
                ax5.set_yticklabels(combined.index, fontsize=9)
                ax5.axvline(0, color='gray', linestyle='--', linewidth=1)
                
                ax5.set_xlabel('Total P&L (₹K)', fontsize=10)
                ax5.set_title(f'Top/Bottom {n_show} Stocks', fontsize=11, fontweight='bold')
            
            ax5.grid(True, alpha=0.3, axis='x')
        
        # ----- 6. Trade Stats Summary -----
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        if 'pnl' in trades.columns:
            wins = trades[trades['pnl'] > 0]
            losses = trades[trades['pnl'] <= 0]
            
            total_trades = len(trades)
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = trades['pnl'].sum()
            avg_pnl = trades['pnl'].mean()
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
            best_trade = trades['pnl'].max()
            worst_trade = trades['pnl'].min()
            
            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
            if np.isinf(profit_factor):
                profit_factor_str = "∞"
            else:
                profit_factor_str = f"{profit_factor:.2f}"
            
            stats_text = f"""
TRADE STATISTICS
{'─' * 35}

Total Trades:     {total_trades:>10,}
Winning Trades:   {win_count:>10,}
Losing Trades:    {loss_count:>10,}

Win Rate:         {win_rate:>10.1f}%

Total P&L:        ₹{total_pnl/100000:>9.2f}L
Avg P&L/Trade:    ₹{avg_pnl/1000:>9.2f}K

Avg Win:          ₹{avg_win/1000:>9.2f}K
Avg Loss:         ₹{avg_loss/1000:>9.2f}K

Best Trade:       ₹{best_trade/1000:>9.2f}K
Worst Trade:      ₹{worst_trade/1000:>9.2f}K

Profit Factor:    {profit_factor_str:>10}
{'─' * 35}
"""
            ax6.text(
                0.1, 0.95, stats_text,
                transform=ax6.transAxes,
                fontsize=10,
                fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            )
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
        
        return fig
    
    # =========================================================================
    # HELPER METHODS
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
    
    def _format_currency(self, x: float, pos: int) -> str:
        """Format number in Indian currency style."""
        if abs(x) >= 1e7:
            return f'₹{x/1e7:.1f}Cr'
        elif abs(x) >= 1e5:
            return f'₹{x/1e5:.1f}L'
        elif abs(x) >= 1e3:
            return f'₹{x/1e3:.0f}K'
        else:
            return f'₹{x:.0f}'
    
    def _format_date_axis(self, ax):
        """Format x-axis for dates."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _save_figure(self, fig: plt.Figure, path: str):
        """Save figure to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"✅ Saved: {path}")


# =============================================================================
# FACTOR PLOTTER
# =============================================================================

class FactorPlotter:
    """
    Factor analysis and feature importance visualization.
    
    Generates:
        - Feature importance plots
        - Factor correlation heatmaps
        - Factor returns analysis
        - SHAP summary plots
    
    Example:
        >>> plotter = FactorPlotter()
        >>> fig = plotter.plot_feature_importance(importance_df)
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150
    ):
        """Initialize factor plotter."""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'positive': '#28A745',
            'negative': '#DC3545',
            'neutral': '#6C757D'
        }
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance (SHAP Values)",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # Sort and get top features
        top_features = importance_df.nlargest(top_n, 'importance')
        
        # ----- 1. Horizontal Bar Chart -----
        ax1 = axes[0]
        
        # Color gradient based on importance
        norm_importance = top_features['importance'] / top_features['importance'].max()
        colors = plt.cm.viridis(norm_importance.values)
        
        y_pos = range(len(top_features))
        bars = ax1.barh(y_pos, top_features['importance'].values,
                       color=colors, alpha=0.85, edgecolor='white')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontsize=11)
        ax1.set_title(f'Top {top_n} Features', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Value labels
        for bar, val in zip(bars, top_features['importance'].values):
            ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8, color='gray')
        
        # ----- 2. Cumulative Importance -----
        ax2 = axes[1]
        
        sorted_imp = importance_df.sort_values('importance', ascending=False)
        cumulative = sorted_imp['importance'].cumsum() / sorted_imp['importance'].sum() * 100
        
        ax2.plot(range(1, len(cumulative) + 1), cumulative.values,
                color=self.colors['primary'], linewidth=2.5)
        ax2.fill_between(range(1, len(cumulative) + 1), cumulative.values,
                        alpha=0.2, color=self.colors['primary'])
        
        # Threshold lines
        ax2.axhline(80, color=self.colors['positive'], linestyle='--',
                   linewidth=1.5, label='80% threshold')
        ax2.axhline(95, color=self.colors['negative'], linestyle='--',
                   linewidth=1.5, label='95% threshold')
        
        # Find crossing points
        idx_80 = np.argmax(cumulative.values >= 80) + 1 if (cumulative.values >= 80).any() else len(cumulative)
        idx_95 = np.argmax(cumulative.values >= 95) + 1 if (cumulative.values >= 95).any() else len(cumulative)
        
        ax2.axvline(idx_80, color=self.colors['positive'], linestyle=':', alpha=0.7)
        ax2.axvline(idx_95, color=self.colors['negative'], linestyle=':', alpha=0.7)
        
        ax2.annotate(f'{idx_80} features\nfor 80%',
                    xy=(idx_80, 80), xytext=(idx_80 + 3, 70),
                    fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))
        
        ax2.set_xlabel('Number of Features', fontsize=11)
        ax2.set_ylabel('Cumulative Importance (%)', fontsize=11)
        ax2.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_xlim(0, min(50, len(cumulative)))
        ax2.set_ylim(0, 102)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_factor_correlation(
        self,
        factor_data: pd.DataFrame,
        title: str = "Factor Correlation Matrix",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot factor correlation heatmap.
        
        Args:
            factor_data: DataFrame with factors as columns
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        corr = factor_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        # Heatmap
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            ax=ax,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            annot_kws={'size': 8}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_factor_returns(
        self,
        factor_returns: pd.DataFrame,
        title: str = "Factor Returns Analysis",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot factor returns analysis.
        
        Args:
            factor_returns: DataFrame with factor returns (columns = factors)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # ----- 1. Cumulative Returns -----
        ax1 = axes[0, 0]
        cumulative = (1 + factor_returns).cumprod()
        
        for i, col in enumerate(cumulative.columns[:10]):
            ax1.plot(cumulative.index, cumulative[col],
                    linewidth=1.5, label=col, alpha=0.85)
        
        ax1.axhline(1, color='gray', linestyle='--', linewidth=1)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Cumulative Return', fontsize=10)
        ax1.set_title('Cumulative Factor Returns', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # ----- 2. Sharpe Ratios -----
        ax2 = axes[0, 1]
        sharpe = (factor_returns.mean() / factor_returns.std() * np.sqrt(252)).sort_values()
        
        colors = [self.colors['positive'] if x > 0 else self.colors['negative']
                 for x in sharpe.values]
        
        ax2.barh(range(len(sharpe)), sharpe.values, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(sharpe)))
        ax2.set_yticklabels(sharpe.index, fontsize=9)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax2.axvline(1, color=self.colors['positive'], linestyle=':', alpha=0.7)
        ax2.set_xlabel('Sharpe Ratio', fontsize=10)
        ax2.set_title('Factor Sharpe Ratios (Ann.)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # ----- 3. Volatility -----
        ax3 = axes[1, 0]
        vol = (factor_returns.std() * np.sqrt(252) * 100).sort_values()
        
        ax3.barh(range(len(vol)), vol.values, color=self.colors['primary'], alpha=0.8)
        ax3.set_yticks(range(len(vol)))
        ax3.set_yticklabels(vol.index, fontsize=9)
        ax3.set_xlabel('Ann. Volatility (%)', fontsize=10)
        ax3.set_title('Factor Volatility', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # ----- 4. Correlation Heatmap -----
        ax4 = axes[1, 1]
        corr = factor_returns.corr()
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   center=0, ax=ax4, annot_kws={'size': 7},
                   cbar_kws={'shrink': 0.8})
        ax4.set_title('Factor Correlations', fontsize=11, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig


# =============================================================================
# RISK PLOTTER
# =============================================================================

class RiskPlotter:
    """
    Risk visualization and analysis.
    
    Generates:
        - VaR and CVaR analysis
        - Risk metrics dashboard
        - Correlation analysis
        - Stress testing visualizations
    
    Example:
        >>> plotter = RiskPlotter()
        >>> fig = plotter.plot_var_analysis(returns)
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 8),
        dpi: int = 150
    ):
        """Initialize risk plotter."""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'positive': '#28A745',
            'negative': '#DC3545',
            'warning': '#FFC107',
            'neutral': '#6C757D'
        }
    
    def plot_var_analysis(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        title: str = "Value at Risk (VaR) Analysis",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot VaR and CVaR analysis.
        
        Args:
            returns: Daily returns series
            confidence_levels: Confidence levels for VaR
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        returns_clean = returns.dropna()
        
        # ----- 1. Distribution with VaR Lines -----
        ax1 = axes[0, 0]
        
        returns_pct = returns_clean * 100
        var_95 = np.percentile(returns_pct, 5)
        var_99 = np.percentile(returns_pct, 1)
        
        # Histogram
        n, bins, patches = ax1.hist(returns_pct, bins=50, density=True,
                                    alpha=0.7, color=self.colors['primary'],
                                    edgecolor='white')
        
        # Color tail red
        for i, patch in enumerate(patches):
            if bins[i] < var_95:
                patch.set_facecolor(self.colors['negative'])
                patch.set_alpha(0.8)
        
        # VaR lines
        ax1.axvline(var_95, color=self.colors['warning'], linestyle='--', 
                   linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
        ax1.axvline(var_99, color=self.colors['negative'], linestyle='--',
                   linewidth=2, label=f'VaR 99%: {var_99:.2f}%')
        
        ax1.set_xlabel('Daily Return (%)', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.set_title('Distribution with VaR', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ----- 2. Rolling VaR -----
        ax2 = axes[0, 1]
        
        window = min(63, len(returns_clean) // 3)  # 3 months or 1/3 of data
        if window > 10:
            rolling_var_95 = returns_clean.rolling(window).quantile(0.05) * 100
            rolling_var_99 = returns_clean.rolling(window).quantile(0.01) * 100
            
            ax2.plot(rolling_var_95.index, rolling_var_95.values,
                    color=self.colors['warning'], linewidth=1.5, label='VaR 95%')
            ax2.plot(rolling_var_99.index, rolling_var_99.values,
                    color=self.colors['negative'], linewidth=1.5, label='VaR 99%')
            ax2.fill_between(rolling_var_95.index, rolling_var_95.values,
                            rolling_var_99.values, alpha=0.3, color=self.colors['warning'])
        
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('VaR (%)', fontsize=10)
        ax2.set_title(f'Rolling {window}-Day VaR', fontsize=11, fontweight='bold')
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # ----- 3. VaR Breaches -----
        ax3 = axes[1, 0]
        
        var_threshold = np.percentile(returns_clean, 5)
        breaches = returns_clean < var_threshold
        breach_returns = returns_clean[breaches] * 100
        
        if len(breach_returns) > 0:
            ax3.scatter(breach_returns.index, breach_returns.values,
                       color=self.colors['negative'], alpha=0.7, s=50,
                       label=f'Breaches ({len(breach_returns)})')
        ax3.axhline(var_threshold * 100, color=self.colors['warning'],
                   linestyle='--', linewidth=2, label=f'VaR 95%: {var_threshold*100:.2f}%')
        
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylabel('Daily Return (%)', fontsize=10)
        ax3.set_title('VaR 95% Breaches', fontsize=11, fontweight='bold')
        ax3.legend(loc='lower left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # ----- 4. VaR vs CVaR Comparison -----
        ax4 = axes[1, 1]
        
        var_values = []
        cvar_values = []
        labels = ['90%', '95%', '99%']
        
        for conf in [0.90, 0.95, 0.99]:
            var = np.percentile(returns_clean, (1-conf)*100) * 100
            cvar = returns_clean[returns_clean <= np.percentile(returns_clean, (1-conf)*100)].mean() * 100
            var_values.append(abs(var))
            cvar_values.append(abs(cvar))
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, var_values, width, label='VaR',
                       color=self.colors['warning'], alpha=0.8)
        bars2 = ax4.bar(x + width/2, cvar_values, width, label='CVaR (ES)',
                       color=self.colors['negative'], alpha=0.8)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_xlabel('Confidence Level', fontsize=10)
        ax4.set_ylabel('Loss (%)', fontsize=10)
        ax4.set_title('VaR vs CVaR Comparison', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.2f}', ha='center', fontsize=8)
        for bar in bars2:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.2f}', ha='center', fontsize=8)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_risk_dashboard(
        self,
        returns: pd.Series,
        equity_curve: Optional[pd.Series] = None,
        title: str = "Risk Metrics Dashboard",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Comprehensive risk metrics dashboard.
        
        Args:
            returns: Daily returns series
            equity_curve: Optional equity curve for drawdown
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        returns_clean = returns.dropna()
        
        # Calculate metrics
        metrics = self._calculate_risk_metrics(returns_clean)
        
        # ----- Row 1: Key Metrics Cards -----
        ax_cards = fig.add_subplot(gs[0, :])
        ax_cards.axis('off')
        
        card_data = [
            ('Volatility\n(Ann.)', f"{metrics['volatility']*100:.1f}%", self.colors['warning']),
            ('VaR 95%', f"{metrics['var_95']*100:.2f}%", self.colors['negative']),
            ('CVaR 95%', f"{metrics['cvar_95']*100:.2f}%", self.colors['negative']),
            ('Max DD', f"{metrics['max_dd']*100:.1f}%", self.colors['negative']),
            ('Sharpe', f"{metrics['sharpe']:.2f}", self.colors['primary']),
            ('Sortino', f"{metrics['sortino']:.2f}", self.colors['primary']),
        ]
        
        for i, (name, value, color) in enumerate(card_data):
            x = 0.08 + i * 0.15
            
            # Card background
            rect = mpatches.FancyBboxPatch(
                (x - 0.05, 0.15), 0.12, 0.7,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=0.15,
                edgecolor=color, linewidth=2,
                transform=ax_cards.transAxes
            )
            ax_cards.add_patch(rect)
            
            # Value
            ax_cards.text(x + 0.01, 0.6, value, fontsize=18, fontweight='bold',
                         color=color, transform=ax_cards.transAxes,
                         ha='center', va='center')
            # Label
            ax_cards.text(x + 0.01, 0.35, name, fontsize=9,
                         color='gray', transform=ax_cards.transAxes,
                         ha='center', va='center')
        
        # ----- Row 2: Drawdown -----
        ax2 = fig.add_subplot(gs[1, :2])
        
        if equity_curve is not None:
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak * 100
        else:
            cum_returns = (1 + returns_clean).cumprod()
            peak = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns - peak) / peak * 100
        
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color=self.colors['negative'], alpha=0.5)
        ax2.plot(drawdown.index, drawdown.values,
                color=self.colors['negative'], linewidth=1)
        ax2.axhline(0, color='black', linewidth=0.5)
        
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_title('Drawdown Over Time', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ----- Row 2: Rolling Volatility -----
        ax3 = fig.add_subplot(gs[1, 2])
        
        window = min(21, len(returns_clean) // 5)
        if window > 5:
            rolling_vol = returns_clean.rolling(window).std() * np.sqrt(252) * 100
            
            ax3.plot(rolling_vol.index, rolling_vol.values,
                    color=self.colors['warning'], linewidth=1.5)
            ax3.axhline(rolling_vol.mean(), color=self.colors['primary'],
                       linestyle='--', linewidth=1.5)
        
        ax3.set_ylabel('Volatility (%)', fontsize=10)
        ax3.set_title(f'Rolling {window}-Day Volatility', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ----- Row 3: Return vs Vol Scatter -----
        ax4 = fig.add_subplot(gs[2, 0])
        
        monthly_ret = returns_clean.resample('M').apply(lambda x: (1+x).prod() - 1) * 100
        monthly_vol = returns_clean.resample('M').std() * np.sqrt(21) * 100
        
        colors_scatter = [self.colors['positive'] if r > 0 else self.colors['negative']
                         for r in monthly_ret.values]
        ax4.scatter(monthly_vol.values, monthly_ret.values, c=colors_scatter,
                   alpha=0.6, s=50, edgecolor='white')
        ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        ax4.set_xlabel('Monthly Volatility (%)', fontsize=10)
        ax4.set_ylabel('Monthly Return (%)', fontsize=10)
        ax4.set_title('Return vs Volatility', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # ----- Row 3: Return Percentiles -----
        ax5 = fig.add_subplot(gs[2, 1])
        
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = [np.percentile(returns_clean, p) * 100 for p in percentiles]
        
        colors_bar = [self.colors['negative'] if v < 0 else self.colors['positive']
                     for v in pct_values]
        
        ax5.barh(range(len(percentiles)), pct_values, color=colors_bar, alpha=0.8)
        ax5.set_yticks(range(len(percentiles)))
        ax5.set_yticklabels([f'{p}th' for p in percentiles])
        ax5.axvline(0, color='gray', linestyle='--', linewidth=1)
        
        ax5.set_xlabel('Daily Return (%)', fontsize=10)
        ax5.set_title('Return Percentiles', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # ----- Row 3: Stats Table -----
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_text = f"""
RISK STATISTICS
{'─' * 30}

Skewness:       {metrics['skewness']:>10.4f}
Kurtosis:       {metrics['kurtosis']:>10.4f}

Best Day:       {metrics['best_day']*100:>10.2f}%
Worst Day:      {metrics['worst_day']*100:>10.2f}%

Positive Days:  {metrics['positive_pct']*100:>10.1f}%
Avg Win:        {metrics['avg_positive']*100:>10.4f}%
Avg Loss:       {metrics['avg_negative']*100:>10.4f}%

Max DD Duration: {metrics['max_dd_duration']:>9} days
{'─' * 30}
"""
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=9, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate all risk metrics."""
        # Basic
        volatility = returns.std() * np.sqrt(252)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        cvar_99 = returns[returns <= var_99].mean() if (returns <= var_99).any() else var_99
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()
        
        # DD Duration
        is_dd = drawdown < 0
        if is_dd.any():
            dd_groups = (~is_dd).cumsum()
            max_dd_duration = is_dd.groupby(dd_groups).sum().max()
        else:
            max_dd_duration = 0
        
        # Ratios
        ann_return = returns.mean() * 252
        sharpe = ann_return / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        downside = returns[returns < 0]
        sortino = ann_return / (downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0
        
        # Avg positive/negative
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_dd': max_dd,
            'max_dd_duration': int(max_dd_duration) if pd.notna(max_dd_duration) else 0,
            'sharpe': sharpe,
            'sortino': sortino,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_pct': (returns > 0).mean(),
            'avg_positive': positive_returns.mean() if len(positive_returns) > 0 else 0,
            'avg_negative': negative_returns.mean() if len(negative_returns) > 0 else 0
        }


# =============================================================================
# QUICK PLOT FUNCTION
# =============================================================================

def quick_plot_all(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    output_dir: str = "reports/charts",
    show: bool = False
) -> Dict[str, str]:
    """
    Generate all standard plots quickly.
    
    Args:
        equity_curve: Portfolio equity curve
        returns: Daily returns series
        trades: Optional trade history
        feature_importance: Optional feature importance DataFrame
        output_dir: Directory to save plots
        show: Whether to display plots
        
    Returns:
        Dictionary of plot_name -> file_path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    perf = PerformancePlotter()
    factor = FactorPlotter()
    risk = RiskPlotter()
    
    # Performance plots
    plots_config = [
        ('equity_curve', lambda: perf.plot_equity_curve(
            equity_curve, save_path=f"{output_dir}/equity_curve.png", show=show)),
        ('returns_distribution', lambda: perf.plot_returns_distribution(
            returns, save_path=f"{output_dir}/returns_distribution.png", show=show)),
        ('rolling_metrics', lambda: perf.plot_rolling_metrics(
            returns, save_path=f"{output_dir}/rolling_metrics.png", show=show)),
        ('monthly_heatmap', lambda: perf.plot_monthly_heatmap(
            returns, save_path=f"{output_dir}/monthly_heatmap.png", show=show)),
        ('var_analysis', lambda: risk.plot_var_analysis(
            returns, save_path=f"{output_dir}/var_analysis.png", show=show)),
        ('risk_dashboard', lambda: risk.plot_risk_dashboard(
            returns, equity_curve, save_path=f"{output_dir}/risk_dashboard.png", show=show)),
    ]
    
    for name, plot_func in plots_config:
        try:
            plot_func()
            saved_plots[name] = f"{output_dir}/{name}.png"
        except Exception as e:
            print(f"⚠️ Failed: {name} - {e}")
    
    # Optional plots
    if trades is not None and not trades.empty:
        try:
            perf.plot_trade_analysis(
                trades, save_path=f"{output_dir}/trade_analysis.png", show=show)
            saved_plots['trade_analysis'] = f"{output_dir}/trade_analysis.png"
        except Exception as e:
            print(f"⚠️ Failed: trade_analysis - {e}")
    
    if feature_importance is not None and not feature_importance.empty:
        try:
            factor.plot_feature_importance(
                feature_importance, save_path=f"{output_dir}/feature_importance.png", show=show)
            saved_plots['feature_importance'] = f"{output_dir}/feature_importance.png"
        except Exception as e:
            print(f"⚠️ Failed: feature_importance - {e}")
    
    print(f"\n✅ Generated {len(saved_plots)} plots in {output_dir}/")
    return saved_plots