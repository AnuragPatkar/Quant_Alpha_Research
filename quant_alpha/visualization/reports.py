"""
Report Generation Module - Pure Python
========================================
No HTML! Clean Python-based reporting.

Output Formats:
    - PDF (matplotlib multi-page)
    - Terminal (Rich library)
    - JSON (metrics export)
    - Excel (openpyxl)
    - CSV (data export)
    - Pickle (full results)

Classes:
    - ReportGenerator: Main report generation class
    
Functions:
    - generate_report(): Quick report generation
    - print_metrics(): Terminal output
    - save_all_charts(): Save all PNG charts
    - export_to_json(): Export metrics to JSON
    - export_to_excel(): Export to Excel
    - export_to_csv(): Export data to CSV

Author: Senior Quant Team
Last Updated: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

# Optional: Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("Rich library not available. Using simple terminal output.")

# Import plotters
try:
    from .plots import PerformancePlotter, FactorPlotter, RiskPlotter
except ImportError:
    from quant_alpha.visualization.plots import PerformancePlotter, FactorPlotter, RiskPlotter


# =============================================================================
# UTILITIES
# =============================================================================

def validate_backtest_results(results: Dict[str, Any]) -> bool:
    """Validate backtest results structure."""
    required_keys = ['metrics']
    for key in required_keys:
        if key not in results:
            logger.warning(f"Missing required key: {key}")
            return False
    return True


def safe_get(d: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    try:
        return d.get(key, default)
    except (AttributeError, TypeError):
        return default


# =============================================================================
# REPORT GENERATOR CLASS
# =============================================================================

class ReportGenerator:
    """
    Pure Python Report Generator.
    
    Generates professional reports without HTML.
    All outputs are native Python formats.
    
    Supported Formats:
        - PDF: Multi-page matplotlib report
        - Terminal: Rich library beautiful output
        - JSON: Metrics and config export
        - Excel: Multi-sheet workbook
        - CSV: Individual data files
        - PNG: Chart images
        - Pickle: Full data preservation
    
    Example:
        >>> generator = ReportGenerator(output_dir='reports')
        >>> generator.generate_full_report(backtest_results)
        
        # Or specific formats
        >>> generator.generate_pdf(backtest_results)
        >>> generator.print_terminal_report(backtest_results)
        >>> generator.export_to_excel(backtest_results)
    """
    
    def __init__(
        self,
        output_dir: str = "reports",
        style: str = 'default',
        dpi: int = 150,
        currency_symbol: str = '$',
        indian_format: bool = False
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Base directory for all outputs
            style: Plot style ('default', 'dark', 'minimal')
            dpi: Resolution for saved figures
            currency_symbol: Currency symbol
            indian_format: Use Indian formatting
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self.dpi = dpi
        self.currency_symbol = currency_symbol
        self.indian_format = indian_format
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize plotters
        self.perf_plotter = PerformancePlotter(
            style=style, 
            dpi=dpi,
            currency_symbol=currency_symbol,
            indian_format=indian_format
        )
        self.factor_plotter = FactorPlotter(dpi=dpi)
        self.risk_plotter = RiskPlotter(dpi=dpi)
        
        # Rich console
        if RICH_AVAILABLE:
            self.console = Console()
        
        # Colors for terminal
        self.colors = {
            'positive': '#28A745',
            'negative': '#DC3545',
            'warning': '#FFC107',
            'info': '#17A2B8',
            'primary': '#2E86AB'
        }
    
    # =========================================================================
    # MAIN GENERATION METHODS
    # =========================================================================
    
    def generate_full_report(
        self,
        backtest_results: Dict[str, Any],
        formats: List[str] = ['pdf', 'terminal', 'json', 'png'],
        report_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate reports in multiple formats.
        
        Args:
            backtest_results: Complete backtest results dictionary
            formats: List of formats to generate
                     Options: 'pdf', 'terminal', 'json', 'excel', 'csv', 'png', 'pickle'
            report_name: Base name for output files
            
        Returns:
            Dictionary mapping format to file path
        """
        if not validate_backtest_results(backtest_results):
            logger.error("Invalid backtest results structure")
            return {}
        
        if report_name is None:
            report_name = f"backtest_report_{self.timestamp}"
        
        outputs = {}
        
        print(f"\n{'='*60}")
        print(f"  GENERATING REPORTS: {report_name}")
        print(f"{'='*60}\n")
        
        # Generate each format
        if 'pdf' in formats:
            print("📄 Generating PDF report...")
            try:
                outputs['pdf'] = self.generate_pdf(backtest_results, report_name)
            except Exception as e:
                logger.error(f"PDF generation failed: {e}", exc_info=True)
                print(f"⚠️  PDF generation failed: {e}")
        
        if 'terminal' in formats:
            print("💻 Printing terminal report...")
            try:
                self.print_terminal_report(backtest_results)
                outputs['terminal'] = 'Printed to console'
            except Exception as e:
                logger.error(f"Terminal report failed: {e}", exc_info=True)
                print(f"⚠️  Terminal report failed: {e}")
        
        if 'json' in formats:
            print("📋 Exporting to JSON...")
            try:
                outputs['json'] = self.export_to_json(backtest_results, report_name)
            except Exception as e:
                logger.error(f"JSON export failed: {e}", exc_info=True)
                print(f"⚠️  JSON export failed: {e}")
        
        if 'excel' in formats:
            print("📊 Exporting to Excel...")
            try:
                outputs['excel'] = self.export_to_excel(backtest_results, report_name)
            except Exception as e:
                logger.error(f"Excel export failed: {e}", exc_info=True)
                print(f"⚠️  Excel export failed: {e}")
        
        if 'csv' in formats:
            print("📁 Exporting to CSV...")
            try:
                outputs['csv'] = self.export_to_csv(backtest_results, report_name)
            except Exception as e:
                logger.error(f"CSV export failed: {e}", exc_info=True)
                print(f"⚠️  CSV export failed: {e}")
        
        if 'png' in formats:
            print("🖼️  Saving charts...")
            try:
                outputs['png'] = self.save_all_charts(backtest_results, report_name)
            except Exception as e:
                logger.error(f"PNG charts failed: {e}", exc_info=True)
                print(f"⚠️  PNG charts failed: {e}")
        
        if 'pickle' in formats:
            print("💾 Saving pickle...")
            try:
                outputs['pickle'] = self.export_to_pickle(backtest_results, report_name)
            except Exception as e:
                logger.error(f"Pickle export failed: {e}", exc_info=True)
                print(f"⚠️  Pickle export failed: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"  REPORT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Formats generated: {len(outputs)}")
        for fmt, path in outputs.items():
            if path != 'Printed to console':
                print(f"    ✅ {fmt.upper()}: {path}")
            else:
                print(f"    ✅ {fmt.upper()}: {path}")
        print(f"{'='*60}\n")
        
        return outputs
    
    # =========================================================================
    # PDF REPORT
    # =========================================================================
    
    def generate_pdf(
        self,
        backtest_results: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive multi-page PDF report.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Output filename (without extension)
            
        Returns:
            Path to generated PDF
        """
        if report_name is None:
            report_name = f"backtest_report_{self.timestamp}"
        
        pdf_path = self.output_dir / f"{report_name}.pdf"
        
        # Extract data
        metrics = safe_get(backtest_results, 'metrics', {})
        equity_curve = safe_get(backtest_results, 'equity_curve', pd.Series())
        returns = safe_get(backtest_results, 'returns', pd.Series())
        trades = safe_get(backtest_results, 'trades', pd.DataFrame())
        feature_importance = safe_get(backtest_results, 'feature_importance', pd.DataFrame())
        ic_series = safe_get(backtest_results, 'ic_series', pd.Series())
        config = safe_get(backtest_results, 'config', {})
        
        logger.info(f"Generating PDF report: {pdf_path}")
        
        with PdfPages(pdf_path) as pdf:
            
            # Page 1: Title & Executive Summary
            try:
                self._create_title_page(pdf, metrics, config)
            except Exception as e:
                logger.error(f"Title page failed: {e}")
            
            # Page 2: Key Metrics Dashboard
            try:
                self._create_metrics_page(pdf, metrics)
            except Exception as e:
                logger.error(f"Metrics page failed: {e}")
            
            # Page 3: Equity Curve
            if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
                try:
                    fig = self.perf_plotter.plot_equity_curve(
                        equity_curve, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Equity curve plot failed: {e}")
            
            # Page 4: Returns Distribution
            if isinstance(returns, pd.Series) and not returns.empty:
                try:
                    fig = self.perf_plotter.plot_returns_distribution(
                        returns, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Returns distribution plot failed: {e}")
            
            # Page 5: Rolling Metrics
            if isinstance(returns, pd.Series) and not returns.empty and len(returns) >= 63:
                try:
                    fig = self.perf_plotter.plot_rolling_metrics(
                        returns, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Rolling metrics plot failed: {e}")
            
            # Page 6: Monthly Heatmap
            if isinstance(returns, pd.Series) and not returns.empty:
                try:
                    fig = self.perf_plotter.plot_monthly_heatmap(
                        returns, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Monthly heatmap plot failed: {e}")
            
            # Page 7: Trade Analysis
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                try:
                    fig = self.perf_plotter.plot_trade_analysis(
                        trades, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Trade analysis plot failed: {e}")
            
            # Page 8: Feature Importance
            if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
                try:
                    fig = self.factor_plotter.plot_feature_importance(
                        feature_importance, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Feature importance plot failed: {e}")
            
            # Page 9: IC Series
            if isinstance(ic_series, pd.Series) and not ic_series.empty:
                try:
                    fig = self.factor_plotter.plot_ic_series(
                        ic_series, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"IC series plot failed: {e}")
            
            # Page 10: VaR Analysis
            if isinstance(returns, pd.Series) and not returns.empty:
                try:
                    fig = self.risk_plotter.plot_var_analysis(
                        returns, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"VaR analysis plot failed: {e}")
            
            # Page 11: Rolling Volatility
            if isinstance(returns, pd.Series) and not returns.empty and len(returns) >= 21:
                try:
                    fig = self.risk_plotter.plot_rolling_volatility(
                        returns, show=False, close_after_save=False
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Rolling volatility plot failed: {e}")
            
            # Page 12: Trade Details Table
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                try:
                    self._create_trades_table_page(pdf, trades)
                except Exception as e:
                    logger.error(f"Trades table page failed: {e}")
        
        logger.info(f"PDF saved: {pdf_path}")
        print(f"✅ PDF saved: {pdf_path}")
        return str(pdf_path)
    
    def _create_title_page(
        self,
        pdf: PdfPages,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ):
        """Create title page with executive summary."""
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.axis('off')
        
        # Title
        fig.text(0.5, 0.92, 'ML ALPHA MODEL', fontsize=32, fontweight='bold',
                ha='center', va='top', color='#2E86AB')
        fig.text(0.5, 0.87, 'Backtest Report', fontsize=22,
                ha='center', va='top', color='#555555')
        
        # Horizontal line
        line = plt.Line2D([0.1, 0.9], [0.84, 0.84], transform=fig.transFigure,
                         color='#2E86AB', linewidth=2)
        fig.add_artist(line)
        
        # Timestamp
        fig.text(0.5, 0.81, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                fontsize=11, ha='center', va='top', color='#888888')
        
        # Configuration Box
        start_date = safe_get(config, 'start_date', 'N/A')
        end_date = safe_get(config, 'end_date', 'N/A')
        universe = str(safe_get(config, 'universe', 'N/A'))[:40]
        initial_capital = safe_get(config, 'initial_capital', 0)
        strategy = str(safe_get(config, 'strategy', 'ML Multi-Factor'))[:40]
        
        config_text = f"""
┌─────────────────────────────────────────────────────────────┐
│                      CONFIGURATION                          │
├─────────────────────────────────────────────────────────────┤
│  Period:           {str(start_date):>15} → {str(end_date):<15} │
│  Universe:         {universe:>40} │
│  Initial Capital:  {self.currency_symbol}{initial_capital:>38,.0f} │
│  Strategy:         {strategy:>40} │
└─────────────────────────────────────────────────────────────┘
"""
        fig.text(0.5, 0.75, config_text, fontsize=10, ha='center', va='top',
                fontfamily='monospace', color='#333333')
        
        # Key Metrics Summary
        cagr = safe_get(metrics, 'cagr', 0) * 100
        sharpe = safe_get(metrics, 'sharpe_ratio', 0)
        max_dd = safe_get(metrics, 'max_drawdown', 0) * 100
        win_rate = safe_get(metrics, 'win_rate', 0) * 100
        total_return = safe_get(metrics, 'total_return', 0) * 100
        sortino = safe_get(metrics, 'sortino_ratio', 0)
        calmar = safe_get(metrics, 'calmar_ratio', 0)
        volatility = safe_get(metrics, 'volatility', 0) * 100
        var_95 = safe_get(metrics, 'var_95', 0) * 100
        profit_factor = safe_get(metrics, 'profit_factor', 0)
        total_trades = safe_get(metrics, 'total_trades', 0)
        
        metrics_text = f"""
╔═════════════════════════════════════════════════════════════╗
║                    EXECUTIVE SUMMARY                        ║
╠═════════════════════════════════════════════════════════════╣
║                                                             ║
║   RETURNS                          RISK-ADJUSTED            ║
║   ───────────────────────          ─────────────────────    ║
║   Total Return:    {total_return:>8.2f}%         Sharpe Ratio:  {sharpe:>8.2f}   ║
║   CAGR:            {cagr:>8.2f}%         Sortino Ratio: {sortino:>8.2f}   ║
║                                    Calmar Ratio:  {calmar:>8.2f}   ║
║                                                             ║
║   RISK                             TRADING                  ║
║   ───────────────────────          ─────────────────────    ║
║   Max Drawdown:    {max_dd:>8.2f}%         Win Rate:      {win_rate:>8.1f}%  ║
║   Volatility:      {volatility:>8.2f}%         Profit Factor: {profit_factor:>8.2f}   ║
║   VaR (95%):       {var_95:>8.2f}%         Total Trades:  {total_trades:>8.0f}   ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
"""
        fig.text(0.5, 0.55, metrics_text, fontsize=10, ha='center', va='top',
                fontfamily='monospace', color='#333333')
        
        # Performance Grade
        if cagr > 15 and sharpe > 1.5:
            grade = 'EXCELLENT'
            grade_color = '#28A745'
            grade_desc = 'Strategy shows outstanding risk-adjusted returns'
        elif cagr > 10 and sharpe > 1.0:
            grade = 'GOOD'
            grade_color = '#2E86AB'
            grade_desc = 'Strategy performs well with acceptable risk'
        elif cagr > 5 and sharpe > 0.5:
            grade = 'MODERATE'
            grade_color = '#FFC107'
            grade_desc = 'Strategy needs optimization'
        else:
            grade = 'NEEDS IMPROVEMENT'
            grade_color = '#DC3545'
            grade_desc = 'Strategy requires significant changes'
        
        # Grade Box
        grade_box = mpatches.FancyBboxPatch(
            (0.25, 0.15), 0.5, 0.12,
            boxstyle="round,pad=0.02",
            facecolor=grade_color,
            alpha=0.15,
            edgecolor=grade_color,
            linewidth=3,
            transform=fig.transFigure
        )
        fig.patches.append(grade_box)
        
        fig.text(0.5, 0.22, f"Overall Rating: {grade}",
                fontsize=18, fontweight='bold', ha='center', color=grade_color)
        fig.text(0.5, 0.17, grade_desc,
                fontsize=11, ha='center', color='#555555')
        
        # Footer
        fig.text(0.5, 0.03, 'Quant Alpha Framework v1.0 | ML-Based Multi-Factor Trading',
                fontsize=9, ha='center', color='#888888')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_metrics_page(
        self,
        pdf: PdfPages,
        metrics: Dict[str, float]
    ):
        """Create detailed metrics page."""
        fig = plt.figure(figsize=(14, 10))
        
        # Title
        fig.text(0.5, 0.95, 'Detailed Performance Metrics',
                fontsize=18, fontweight='bold', ha='center', color='#2E86AB')
        
        # Create metric cards layout
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3,
                     top=0.88, bottom=0.1, left=0.05, right=0.95)
        
        # Define metrics for display
        metric_cards = [
            # Row 1: Return Metrics
            ('CAGR', f"{safe_get(metrics, 'cagr', 0)*100:.2f}%", '#2E86AB'),
            ('Total Return', f"{safe_get(metrics, 'total_return', 0)*100:.2f}%", '#2E86AB'),
            ('Avg Daily', f"{safe_get(metrics, 'avg_daily_return', 0)*100:.4f}%", '#2E86AB'),
            ('Best Day', f"{safe_get(metrics, 'best_day', 0)*100:.2f}%", '#28A745'),
            
            # Row 2: Risk Metrics
            ('Max Drawdown', f"{safe_get(metrics, 'max_drawdown', 0)*100:.2f}%", '#DC3545'),
            ('Volatility', f"{safe_get(metrics, 'volatility', 0)*100:.2f}%", '#FFC107'),
            ('VaR (95%)', f"{safe_get(metrics, 'var_95', 0)*100:.2f}%", '#DC3545'),
            ('Worst Day', f"{safe_get(metrics, 'worst_day', 0)*100:.2f}%", '#DC3545'),
            
            # Row 3: Risk-Adjusted
            ('Sharpe Ratio', f"{safe_get(metrics, 'sharpe_ratio', 0):.2f}", '#2E86AB'),
            ('Sortino Ratio', f"{safe_get(metrics, 'sortino_ratio', 0):.2f}", '#2E86AB'),
            ('Calmar Ratio', f"{safe_get(metrics, 'calmar_ratio', 0):.2f}", '#2E86AB'),
            ('Info Ratio', f"{safe_get(metrics, 'information_ratio', 0):.2f}", '#2E86AB'),
            
            # Row 4: Trading
            ('Win Rate', f"{safe_get(metrics, 'win_rate', 0)*100:.1f}%", '#28A745'),
            ('Profit Factor', f"{safe_get(metrics, 'profit_factor', 0):.2f}", '#28A745'),
            ('Total Trades', f"{safe_get(metrics, 'total_trades', 0):.0f}", '#6C757D'),
            ('Avg Trade', f"{safe_get(metrics, 'avg_trade_return', 0)*100:.3f}%", '#6C757D'),
        ]
        
        for i, (name, value, color) in enumerate(metric_cards):
            row = i // 4
            col = i % 4
            
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
            
            # Card background
            rect = mpatches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.05",
                facecolor=color,
                alpha=0.1,
                edgecolor=color,
                linewidth=2,
                transform=ax.transAxes
            )
            ax.add_patch(rect)
            
            # Value
            ax.text(0.5, 0.6, value, fontsize=20, fontweight='bold',
                   ha='center', va='center', color=color,
                   transform=ax.transAxes)
            
            # Label
            ax.text(0.5, 0.25, name, fontsize=10, ha='center', va='center',
                   color='#555555', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_trades_table_page(
        self,
        pdf: PdfPages,
        trades: pd.DataFrame,
        n_rows: int = 30
    ):
        """Create trades table page."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Title
        fig.text(0.5, 0.95, f'Recent Trades (Last {min(n_rows, len(trades))})',
                fontsize=16, fontweight='bold', ha='center', color='#2E86AB')
        
        # Prepare data
        recent_trades = trades.tail(n_rows).copy()
        
        # Sort by date if available
        if 'date' in recent_trades.columns:
            recent_trades = recent_trades.sort_values('date', ascending=False)
        elif 'exit_date' in recent_trades.columns:
            recent_trades = recent_trades.sort_values('exit_date', ascending=False)
        
        # Select columns
        display_cols = []
        col_widths = []
        
        possible_cols = {
            'date': 0.15,
            'exit_date': 0.15,
            'ticker': 0.12,
            'symbol': 0.12,
            'side': 0.08,
            'pnl': 0.15,
            'return_pct': 0.12,
            'quantity': 0.10
        }
        
        for col, width in possible_cols.items():
            if col in recent_trades.columns:
                display_cols.append(col)
                col_widths.append(width)
                if len(display_cols) >= 6:  # Limit columns
                    break
        
        if display_cols:
            table_data = recent_trades[display_cols].head(n_rows).values.tolist()
            
            # Format data
            formatted_data = []
            for row in table_data:
                formatted_row = []
                for i, val in enumerate(row):
                    col = display_cols[i]
                    if col == 'pnl':
                        formatted_row.append(f"{self.currency_symbol}{val:,.0f}")
                    elif col == 'return_pct':
                        formatted_row.append(f"{val*100:.2f}%")
                    elif col in ['date', 'exit_date']:
                        formatted_row.append(str(val)[:10])
                    elif pd.isna(val):
                        formatted_row.append('-')
                    else:
                        formatted_row.append(str(val))
                formatted_data.append(formatted_row)
            
            # Create table
            table = ax.table(
                cellText=formatted_data,
                colLabels=[c.replace('_', ' ').upper() for c in display_cols],
                cellLoc='center',
                loc='center',
                colWidths=col_widths
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            # Style header
            for j in range(len(display_cols)):
                table[(0, j)].set_facecolor('#2E86AB')
                table[(0, j)].set_text_props(color='white', fontweight='bold')
            
            # Color P&L cells
            if 'pnl' in display_cols:
                pnl_col = display_cols.index('pnl')
                for i, row in enumerate(table_data, 1):
                    try:
                        if row[pnl_col] > 0:
                            table[(i, pnl_col)].set_text_props(color='#28A745', weight='bold')
                        else:
                            table[(i, pnl_col)].set_text_props(color='#DC3545', weight='bold')
                    except:
                        pass
        else:
            ax.text(0.5, 0.5, 'No trade data available',
                   ha='center', va='center', fontsize=14, color='#888888')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # =========================================================================
    # TERMINAL REPORT
    # =========================================================================
    
    def print_terminal_report(self, backtest_results: Dict[str, Any]):
        """
        Print beautiful report in terminal.
        
        Uses Rich library if available, otherwise plain text.
        
        Args:
            backtest_results: Backtest results dictionary
        """
        metrics = safe_get(backtest_results, 'metrics', {})
        config = safe_get(backtest_results, 'config', {})
        trades = safe_get(backtest_results, 'trades', pd.DataFrame())
        
        if RICH_AVAILABLE:
            self._print_rich_report(metrics, config, trades)
        else:
            self._print_simple_report(metrics, config, trades)
    
    def _print_rich_report(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        trades: pd.DataFrame
    ):
        """Print report using Rich library."""
        console = self.console
        
        # Header Panel
        console.print()
        header_text = Text()
        header_text.append("📊 ML ALPHA MODEL ", style="bold cyan")
        header_text.append("- Backtest Report", style="white")
        console.print(Panel(header_text, border_style="cyan", padding=(1, 2)))
        
        # Configuration Table
        config_table = Table(
            title="⚙️ Configuration",
            box=box.ROUNDED,
            title_style="bold cyan",
            border_style="cyan"
        )
        config_table.add_column("Parameter", style="cyan", width=20)
        config_table.add_column("Value", style="white", width=40)
        
        config_table.add_row("Period", f"{safe_get(config, 'start_date', 'N/A')} → {safe_get(config, 'end_date', 'N/A')}")
        config_table.add_row("Universe", str(safe_get(config, 'universe', 'N/A')))
        config_table.add_row("Initial Capital", f"{self.currency_symbol}{safe_get(config, 'initial_capital', 0):,.0f}")
        config_table.add_row("Generated", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        console.print(config_table)
        console.print()
        
        # Performance Metrics Table
        perf_table = Table(
            title="🎯 Performance Metrics",
            box=box.ROUNDED,
            title_style="bold cyan",
            border_style="cyan"
        )
        perf_table.add_column("Metric", style="cyan", width=22)
        perf_table.add_column("Value", style="white", justify="right", width=15)
        perf_table.add_column("Status", justify="center", width=10)
        
        # Add metrics with status
        def get_status(value, good_threshold, bad_threshold=None, higher_is_better=True):
            if higher_is_better:
                if value >= good_threshold:
                    return "[green]●[/green]"
                elif bad_threshold and value < bad_threshold:
                    return "[red]●[/red]"
                else:
                    return "[yellow]●[/yellow]"
            else:
                if value <= good_threshold:
                    return "[green]●[/green]"
                elif bad_threshold and value > bad_threshold:
                    return "[red]●[/red]"
                else:
                    return "[yellow]●[/yellow]"
        
        cagr = safe_get(metrics, 'cagr', 0)
        perf_table.add_row(
            "CAGR",
            f"{cagr*100:.2f}%",
            get_status(cagr, 0.12, 0)
        )
        
        total_ret = safe_get(metrics, 'total_return', 0)
        perf_table.add_row(
            "Total Return",
            f"{total_ret*100:.2f}%",
            get_status(total_ret, 0.20, 0)
        )
        
        sharpe = safe_get(metrics, 'sharpe_ratio', 0)
        perf_table.add_row(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            get_status(sharpe, 1.0, 0.5)
        )
        
        sortino = safe_get(metrics, 'sortino_ratio', 0)
        perf_table.add_row(
            "Sortino Ratio",
            f"{sortino:.2f}",
            get_status(sortino, 1.5, 0.5)
        )
        
        max_dd = safe_get(metrics, 'max_drawdown', 0)
        perf_table.add_row(
            "Max Drawdown",
            f"{max_dd*100:.2f}%",
            get_status(abs(max_dd), 0.10, 0.25, higher_is_better=False)
        )
        
        vol = safe_get(metrics, 'volatility', 0)
        perf_table.add_row(
            "Volatility (Ann.)",
            f"{vol*100:.2f}%",
            get_status(vol, 0.15, 0.30, higher_is_better=False)
        )
        
        calmar = safe_get(metrics, 'calmar_ratio', 0)
        perf_table.add_row(
            "Calmar Ratio",
            f"{calmar:.2f}",
            get_status(calmar, 1.0, 0.3)
        )
        
        console.print(perf_table)
        console.print()
        
        # Trading Metrics Table
        trade_table = Table(
            title="💹 Trading Metrics",
            box=box.ROUNDED,
            title_style="bold cyan",
            border_style="cyan"
        )
        trade_table.add_column("Metric", style="cyan", width=22)
        trade_table.add_column("Value", style="white", justify="right", width=15)
        
        win_rate = safe_get(metrics, 'win_rate', 0)
        trade_table.add_row("Win Rate", f"{win_rate*100:.1f}%")
        trade_table.add_row("Profit Factor", f"{safe_get(metrics, 'profit_factor', 0):.2f}")
        trade_table.add_row("Total Trades", f"{safe_get(metrics, 'total_trades', 0):,.0f}")
        
        if isinstance(trades, pd.DataFrame) and not trades.empty and 'pnl' in trades.columns:
            trade_table.add_row("Total P&L", f"{self.currency_symbol}{trades['pnl'].sum():,.0f}")
            trade_table.add_row("Avg P&L/Trade", f"{self.currency_symbol}{trades['pnl'].mean():,.0f}")
            trade_table.add_row("Best Trade", f"{self.currency_symbol}{trades['pnl'].max():,.0f}")
            trade_table.add_row("Worst Trade", f"{self.currency_symbol}{trades['pnl'].min():,.0f}")
        
        console.print(trade_table)
        console.print()
        
        # Risk Metrics Table
        risk_table = Table(
            title="⚠️ Risk Metrics",
            box=box.ROUNDED,
            title_style="bold cyan",
            border_style="cyan"
        )
        risk_table.add_column("Metric", style="cyan", width=22)
        risk_table.add_column("Value", style="white", justify="right", width=15)
        
        risk_table.add_row("VaR (95%)", f"{safe_get(metrics, 'var_95', 0)*100:.2f}%")
        risk_table.add_row("CVaR (95%)", f"{safe_get(metrics, 'cvar_95', 0)*100:.2f}%")
        risk_table.add_row("Skewness", f"{safe_get(metrics, 'skewness', 0):.4f}")
        risk_table.add_row("Kurtosis", f"{safe_get(metrics, 'kurtosis', 0):.4f}")
        risk_table.add_row("Beta", f"{safe_get(metrics, 'beta', 0):.2f}")
        risk_table.add_row("Alpha (Ann.)", f"{safe_get(metrics, 'alpha', 0)*100:.2f}%")
        
        console.print(risk_table)
        console.print()
        
        # Overall Assessment Panel
        cagr_val = safe_get(metrics, 'cagr', 0)
        sharpe_val = safe_get(metrics, 'sharpe_ratio', 0)
        
        if cagr_val > 0.15 and sharpe_val > 1.5:
            grade = "EXCELLENT PERFORMANCE"
            grade_style = "bold green"
            message = "Strategy shows outstanding risk-adjusted returns. Ready for deployment."
        elif cagr_val > 0.10 and sharpe_val > 1.0:
            grade = "GOOD PERFORMANCE"
            grade_style = "bold blue"
            message = "Strategy performs well. Consider minor optimizations."
        elif cagr_val > 0.05:
            grade = "MODERATE PERFORMANCE"
            grade_style = "bold yellow"
            message = "Strategy needs improvement. Review factors and risk management."
        else:
            grade = "NEEDS IMPROVEMENT"
            grade_style = "bold red"
            message = "Strategy underperforming. Significant changes required."
        
        assessment_text = Text()
        assessment_text.append(f"\n{grade}\n\n", style=grade_style)
        assessment_text.append(message, style="white")
        
        console.print(Panel(
            assessment_text,
            title="📈 Assessment",
            border_style=grade_style.replace("bold ", ""),
            padding=(1, 2)
        ))
        console.print()
    
    def _print_simple_report(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        trades: pd.DataFrame
    ):
        """Print simple text report without Rich."""
        print("\n" + "=" * 70)
        print("           ML ALPHA MODEL - BACKTEST REPORT")
        print("=" * 70)
        
        print(f"\n📅 Period: {safe_get(config, 'start_date', 'N/A')} → {safe_get(config, 'end_date', 'N/A')}")
        print(f"🏛️ Universe: {safe_get(config, 'universe', 'N/A')}")
        print(f"💰 Initial Capital: {self.currency_symbol}{safe_get(config, 'initial_capital', 0):,.0f}")
        print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "-" * 70)
        print(" PERFORMANCE METRICS")
        print("-" * 70)
        
        print(f"\n  {'RETURNS':<25} {'RISK-ADJUSTED':<25}")
        print(f"  {'-'*23}   {'-'*23}")
        print(f"  CAGR:           {safe_get(metrics, 'cagr', 0)*100:>8.2f}%    Sharpe Ratio:  {safe_get(metrics, 'sharpe_ratio', 0):>8.2f}")
        print(f"  Total Return:   {safe_get(metrics, 'total_return', 0)*100:>8.2f}%    Sortino Ratio: {safe_get(metrics, 'sortino_ratio', 0):>8.2f}")
        print(f"  Volatility:     {safe_get(metrics, 'volatility', 0)*100:>8.2f}%    Calmar Ratio:  {safe_get(metrics, 'calmar_ratio', 0):>8.2f}")
        
        print(f"\n  {'RISK':<25} {'TRADING':<25}")
        print(f"  {'-'*23}   {'-'*23}")
        print(f"  Max Drawdown:   {safe_get(metrics, 'max_drawdown', 0)*100:>8.2f}%    Win Rate:      {safe_get(metrics, 'win_rate', 0)*100:>8.1f}%")
        print(f"  VaR (95%):      {safe_get(metrics, 'var_95', 0)*100:>8.2f}%    Profit Factor: {safe_get(metrics, 'profit_factor', 0):>8.2f}")
        print(f"  CVaR (95%):     {safe_get(metrics, 'cvar_95', 0)*100:>8.2f}%    Total Trades:  {safe_get(metrics, 'total_trades', 0):>8.0f}")
        
        if isinstance(trades, pd.DataFrame) and not trades.empty and 'pnl' in trades.columns:
            print(f"\n  {'TRADE DETAILS':<25}")
            print(f"  {'-'*23}")
            print(f"  Total P&L:      {self.currency_symbol}{trades['pnl'].sum():>12,.0f}")
            print(f"  Avg P&L/Trade:  {self.currency_symbol}{trades['pnl'].mean():>12,.0f}")
            print(f"  Best Trade:     {self.currency_symbol}{trades['pnl'].max():>12,.0f}")
            print(f"  Worst Trade:    {self.currency_symbol}{trades['pnl'].min():>12,.0f}")
        
        print("\n" + "=" * 70 + "\n")
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def export_to_json(
        self,
        backtest_results: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Export metrics and config to JSON.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Output filename
            
        Returns:
            Path to JSON file
        """
        if report_name is None:
            report_name = f"metrics_{self.timestamp}"
        
        json_path = self.output_dir / f"{report_name}.json"
        
        # Prepare export data
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'config': safe_get(backtest_results, 'config', {}),
            'metrics': safe_get(backtest_results, 'metrics', {}),
            'summary': {}
        }
        
        # Add summary data
        equity = safe_get(backtest_results, 'equity_curve', pd.Series())
        if isinstance(equity, pd.Series) and not equity.empty:
            export_data['summary']['start_date'] = str(equity.index[0])
            export_data['summary']['end_date'] = str(equity.index[-1])
            export_data['summary']['start_value'] = float(equity.iloc[0])
            export_data['summary']['end_value'] = float(equity.iloc[-1])
        
        trades = safe_get(backtest_results, 'trades', pd.DataFrame())
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            export_data['summary']['total_trades'] = len(trades)
            if 'pnl' in trades.columns:
                export_data['summary']['total_pnl'] = float(trades['pnl'].sum())
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        export_data = convert_types(export_data)
        
        # Save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"JSON saved: {json_path}")
        print(f"✅ JSON saved: {json_path}")
        return str(json_path)
    
    def export_to_excel(
        self,
        backtest_results: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Export to Excel with multiple sheets.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Output filename
            
        Returns:
            Path to Excel file
        """
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl not installed")
            print("❌ openpyxl not installed. Run: pip install openpyxl")
            return ""
        
        if report_name is None:
            report_name = f"backtest_{self.timestamp}"
        
        excel_path = self.output_dir / f"{report_name}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Sheet 1: Metrics
            metrics = safe_get(backtest_results, 'metrics', {})
            if metrics:
                metrics_df = pd.DataFrame([
                    {'Metric': k, 'Value': v} for k, v in metrics.items()
                ])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Sheet 2: Configuration
            config = safe_get(backtest_results, 'config', {})
            if config:
                config_df = pd.DataFrame([
                    {'Parameter': k, 'Value': str(v)} for k, v in config.items()
                ])
                config_df.to_excel(writer, sheet_name='Config', index=False)
            
            # Sheet 3: Trades
            trades = safe_get(backtest_results, 'trades', pd.DataFrame())
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                trades.to_excel(writer, sheet_name='Trades', index=False)
            
            # Sheet 4: Returns
            returns = safe_get(backtest_results, 'returns', pd.Series())
            if isinstance(returns, pd.Series) and not returns.empty:
                returns_df = returns.to_frame('daily_return')
                returns_df['cumulative_return'] = (1 + returns).cumprod() - 1
                returns_df.to_excel(writer, sheet_name='Returns')
            
            # Sheet 5: Equity Curve
            equity = safe_get(backtest_results, 'equity_curve', pd.Series())
            if isinstance(equity, pd.Series) and not equity.empty:
                equity_df = equity.to_frame('portfolio_value')
                equity_df.to_excel(writer, sheet_name='Equity')
            
            # Sheet 6: Feature Importance
            feature_imp = safe_get(backtest_results, 'feature_importance', pd.DataFrame())
            if isinstance(feature_imp, pd.DataFrame) and not feature_imp.empty:
                feature_imp.to_excel(writer, sheet_name='Features', index=False)
        
        logger.info(f"Excel saved: {excel_path}")
        print(f"✅ Excel saved: {excel_path}")
        return str(excel_path)
    
    def export_to_csv(
        self,
        backtest_results: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Export data to multiple CSV files.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Base name for files
            
        Returns:
            Path to CSV directory
        """
        if report_name is None:
            report_name = f"backtest_{self.timestamp}"
        
        csv_dir = self.output_dir / f"{report_name}_csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        metrics = safe_get(backtest_results, 'metrics', {})
        if metrics:
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            metrics_df.index.name = 'Metric'
            metrics_df.to_csv(csv_dir / 'metrics.csv')
        
        # Trades
        trades = safe_get(backtest_results, 'trades', pd.DataFrame())
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            trades.to_csv(csv_dir / 'trades.csv', index=False)
        
        # Returns
        returns = safe_get(backtest_results, 'returns', pd.Series())
        if isinstance(returns, pd.Series) and not returns.empty:
            returns.to_frame('daily_return').to_csv(csv_dir / 'returns.csv')
        
        # Equity
        equity = safe_get(backtest_results, 'equity_curve', pd.Series())
        if isinstance(equity, pd.Series) and not equity.empty:
            equity.to_frame('portfolio_value').to_csv(csv_dir / 'equity.csv')
        
        # Features
        feature_imp = safe_get(backtest_results, 'feature_importance', pd.DataFrame())
        if isinstance(feature_imp, pd.DataFrame) and not feature_imp.empty:
            feature_imp.to_csv(csv_dir / 'features.csv', index=False)
        
        logger.info(f"CSV files saved: {csv_dir}")
        print(f"✅ CSV files saved: {csv_dir}")
        return str(csv_dir)
    
    def export_to_pickle(
        self,
        backtest_results: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Export full results to pickle file.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Output filename
            
        Returns:
            Path to pickle file
        """
        if report_name is None:
            report_name = f"backtest_{self.timestamp}"
        
        pkl_path = self.output_dir / f"{report_name}.pkl"
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(backtest_results, f)
        
        logger.info(f"Pickle saved: {pkl_path}")
        print(f"✅ Pickle saved: {pkl_path}")
        return str(pkl_path)
    
    def save_all_charts(
        self,
        backtest_results: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Save all charts as PNG files.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Base name for directory
            
        Returns:
            Path to charts directory
        """
        if report_name is None:
            report_name = f"charts_{self.timestamp}"
        
        charts_dir = self.output_dir / report_name
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        equity = safe_get(backtest_results, 'equity_curve', pd.Series())
        returns = safe_get(backtest_results, 'returns', pd.Series())
        trades = safe_get(backtest_results, 'trades', pd.DataFrame())
        feature_imp = safe_get(backtest_results, 'feature_importance', pd.DataFrame())
        ic_series = safe_get(backtest_results, 'ic_series', pd.Series())
        
        saved = []
        
        # Generate charts
        if isinstance(equity, pd.Series) and not equity.empty:
            try:
                fig = self.perf_plotter.plot_equity_curve(
                    equity, save_path=str(charts_dir / 'equity_curve.png'), 
                    show=False, close_after_save=True
                )
                saved.append('equity_curve.png')
            except Exception as e:
                logger.error(f"equity_curve failed: {e}")
        
        if isinstance(returns, pd.Series) and not returns.empty:
            try:
                fig = self.perf_plotter.plot_returns_distribution(
                    returns, save_path=str(charts_dir / 'returns_distribution.png'), 
                    show=False, close_after_save=True
                )
                saved.append('returns_distribution.png')
            except Exception as e:
                logger.error(f"returns_distribution failed: {e}")
            
            if len(returns) >= 63:
                try:
                    fig = self.perf_plotter.plot_rolling_metrics(
                        returns, save_path=str(charts_dir / 'rolling_metrics.png'), 
                        show=False, close_after_save=True
                    )
                    saved.append('rolling_metrics.png')
                except Exception as e:
                    logger.error(f"rolling_metrics failed: {e}")
            
            try:
                fig = self.perf_plotter.plot_monthly_heatmap(
                    returns, save_path=str(charts_dir / 'monthly_heatmap.png'), 
                    show=False, close_after_save=True
                )
                saved.append('monthly_heatmap.png')
            except Exception as e:
                logger.error(f"monthly_heatmap failed: {e}")
            
            try:
                fig = self.risk_plotter.plot_var_analysis(
                    returns, save_path=str(charts_dir / 'var_analysis.png'), 
                    show=False, close_after_save=True
                )
                saved.append('var_analysis.png')
            except Exception as e:
                logger.error(f"var_analysis failed: {e}")
            
            if len(returns) >= 21:
                try:
                    fig = self.risk_plotter.plot_rolling_volatility(
                        returns, save_path=str(charts_dir / 'rolling_volatility.png'), 
                        show=False, close_after_save=True
                    )
                    saved.append('rolling_volatility.png')
                except Exception as e:
                    logger.error(f"rolling_volatility failed: {e}")
        
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            try:
                fig = self.perf_plotter.plot_trade_analysis(
                    trades, save_path=str(charts_dir / 'trade_analysis.png'), 
                    show=False, close_after_save=True
                )
                saved.append('trade_analysis.png')
            except Exception as e:
                logger.error(f"trade_analysis failed: {e}")
        
        if isinstance(feature_imp, pd.DataFrame) and not feature_imp.empty:
            try:
                fig = self.factor_plotter.plot_feature_importance(
                    feature_imp, save_path=str(charts_dir / 'feature_importance.png'), 
                    show=False, close_after_save=True
                )
                saved.append('feature_importance.png')
            except Exception as e:
                logger.error(f"feature_importance failed: {e}")
        
        if isinstance(ic_series, pd.Series) and not ic_series.empty:
            try:
                fig = self.factor_plotter.plot_ic_series(
                    ic_series, save_path=str(charts_dir / 'ic_series.png'), 
                    show=False, close_after_save=True
                )
                saved.append('ic_series.png')
            except Exception as e:
                logger.error(f"ic_series failed: {e}")
        
        logger.info(f"{len(saved)} charts saved: {charts_dir}")
        print(f"✅ {len(saved)} charts saved: {charts_dir}")
        return str(charts_dir)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_report(
    backtest_results: Dict[str, Any],
    output_dir: str = "reports",
    formats: List[str] = ['pdf', 'terminal', 'json'],
    report_name: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Quick function to generate reports.
    
    Args:
        backtest_results: Backtest results dictionary
        output_dir: Output directory
        formats: List of formats ('pdf', 'terminal', 'json', 'excel', 'csv', 'png', 'pickle')
        report_name: Base name for files
        **kwargs: Additional ReportGenerator arguments
        
    Returns:
        Dictionary mapping format to file path
    
    Example:
        >>> outputs = generate_report(results, formats=['pdf', 'json'])
    """
    generator = ReportGenerator(output_dir, **kwargs)
    return generator.generate_full_report(backtest_results, formats, report_name)


def print_metrics(
    metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Quick function to print metrics to terminal.
    
    Args:
        metrics: Metrics dictionary
        config: Optional config dictionary
        **kwargs: Additional ReportGenerator arguments
    
    Example:
        >>> print_metrics({'cagr': 0.15, 'sharpe_ratio': 1.5})
    """
    generator = ReportGenerator(**kwargs)
    generator.print_terminal_report({
        'metrics': metrics,
        'config': config or {},
        'trades': pd.DataFrame()
    })


def save_all_charts(
    backtest_results: Dict[str, Any],
    output_dir: str = "reports/charts",
    **kwargs
) -> str:
    """
    Quick function to save all charts.
    
    Args:
        backtest_results: Backtest results dictionary
        output_dir: Output directory
        **kwargs: Additional ReportGenerator arguments
        
    Returns:
        Path to charts directory
    
    Example:
        >>> save_all_charts(results, 'my_charts')
    """
    generator = ReportGenerator(output_dir, **kwargs)
    return generator.save_all_charts(backtest_results)


def export_to_json(
    backtest_results: Dict[str, Any],
    output_path: str = "reports/metrics.json",
    **kwargs
) -> str:
    """
    Quick function to export to JSON.
    
    Args:
        backtest_results: Backtest results dictionary
        output_path: Output file path
        **kwargs: Additional ReportGenerator arguments
        
    Returns:
        Path to JSON file
    """
    output_dir = str(Path(output_path).parent)
    filename = Path(output_path).stem
    
    generator = ReportGenerator(output_dir, **kwargs)
    return generator.export_to_json(backtest_results, filename)


def export_to_excel(
    backtest_results: Dict[str, Any],
    output_path: str = "reports/backtest.xlsx",
    **kwargs
) -> str:
    """
    Quick function to export to Excel.
    
    Args:
        backtest_results: Backtest results dictionary
        output_path: Output file path
        **kwargs: Additional ReportGenerator arguments
        
    Returns:
        Path to Excel file
    """
    output_dir = str(Path(output_path).parent)
    filename = Path(output_path).stem
    
    generator = ReportGenerator(output_dir, **kwargs)
    return generator.export_to_excel(backtest_results, filename)


def export_to_csv(
    backtest_results: Dict[str, Any],
    output_dir: str = "reports/csv",
    **kwargs
) -> str:
    """
    Quick function to export to CSV.
    
    Args:
        backtest_results: Backtest results dictionary
        output_dir: Output directory
        **kwargs: Additional ReportGenerator arguments
        
    Returns:
        Path to CSV directory
    """
    generator = ReportGenerator(output_dir, **kwargs)
    return generator.export_to_csv(backtest_results)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    print("Testing reports.py...")
    
    # Create sample backtest results
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.0008, 0.01, len(dates)), index=dates)
    equity = (1 + returns).cumprod() * 100000
    
    # Create sample trades
    trades_data = []
    for i in range(50):
        trades_data.append({
            'date': dates[i*20],
            'ticker': f'STOCK{i%10}',
            'side': 'LONG' if i % 2 == 0 else 'SHORT',
            'pnl': np.random.normal(500, 1000),
            'return_pct': np.random.normal(0.01, 0.02),
            'quantity': 100
        })
    trades_df = pd.DataFrame(trades_data)
    
    # Calculate metrics
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    backtest_results = {
        'equity_curve': equity,
        'returns': returns,
        'trades': trades_df,
        'metrics': {
            'total_return': total_return,
            'cagr': (1 + total_return) ** (252 / len(returns)) - 1,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sharpe * 1.1,
            'max_drawdown': -0.15,
            'volatility': returns.std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'win_rate': 0.58,
            'profit_factor': 1.45,
            'total_trades': len(trades_df),
            'calmar_ratio': total_return / 0.15,
        },
        'config': {
            'start_date': dates[0],
            'end_date': dates[-1],
            'universe': 'NSE Top 100',
            'initial_capital': 100000,
            'strategy': 'ML Multi-Factor'
        }
    }
    
    # Test report generation
    generator = ReportGenerator(output_dir='test_reports')
    
    # Test terminal report
    print("\n=== TERMINAL REPORT ===")
    generator.print_terminal_report(backtest_results)
    
    # Test JSON export
    json_path = generator.export_to_json(backtest_results, 'test_metrics')
    
    # Test full report
    outputs = generator.generate_full_report(
        backtest_results,
        formats=['json', 'csv'],
        report_name='test_full_report'
    )
    
    print("\n✅ All tests passed!")
    print(f"Output files: {outputs}")