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
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

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

# Import plotters
from .plots import PerformancePlotter, FactorPlotter, RiskPlotter


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
        dpi: int = 150
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Base directory for all outputs
            style: Plot style ('default', 'dark', 'minimal')
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self.dpi = dpi
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize plotters
        self.perf_plotter = PerformancePlotter(style=style, dpi=dpi)
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
        if report_name is None:
            report_name = f"backtest_report_{self.timestamp}"
        
        outputs = {}
        
        print(f"\n{'='*60}")
        print(f"  GENERATING REPORTS: {report_name}")
        print(f"{'='*60}\n")
        
        # Generate each format
        if 'pdf' in formats:
            try:
                outputs['pdf'] = self.generate_pdf(backtest_results, report_name)
            except Exception as e:
                print(f"⚠️ PDF generation failed: {e}")
        
        if 'terminal' in formats:
            self.print_terminal_report(backtest_results)
            outputs['terminal'] = 'Printed to console'
        
        if 'json' in formats:
            try:
                outputs['json'] = self.export_to_json(backtest_results, report_name)
            except Exception as e:
                print(f"⚠️ JSON export failed: {e}")
        
        if 'excel' in formats:
            try:
                outputs['excel'] = self.export_to_excel(backtest_results, report_name)
            except Exception as e:
                print(f"⚠️ Excel export failed: {e}")
        
        if 'csv' in formats:
            try:
                outputs['csv'] = self.export_to_csv(backtest_results, report_name)
            except Exception as e:
                print(f"⚠️ CSV export failed: {e}")
        
        if 'png' in formats:
            try:
                outputs['png'] = self.save_all_charts(backtest_results, report_name)
            except Exception as e:
                print(f"⚠️ PNG charts failed: {e}")
        
        if 'pickle' in formats:
            try:
                outputs['pickle'] = self.export_to_pickle(backtest_results, report_name)
            except Exception as e:
                print(f"⚠️ Pickle export failed: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"  REPORT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Formats generated: {len(outputs)}")
        for fmt, path in outputs.items():
            print(f"    ✅ {fmt}: {path}")
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
        metrics = backtest_results.get('metrics', {})
        equity_curve = backtest_results.get('equity_curve', pd.Series())
        returns = backtest_results.get('returns', pd.Series())
        trades = backtest_results.get('trades', pd.DataFrame())
        feature_importance = backtest_results.get('feature_importance', pd.DataFrame())
        config = backtest_results.get('config', {})
        
        print(f"📄 Generating PDF report...")
        
        with PdfPages(pdf_path) as pdf:
            
            # Page 1: Title & Executive Summary
            self._create_title_page(pdf, metrics, config)
            
            # Page 2: Key Metrics Dashboard
            self._create_metrics_page(pdf, metrics)
            
            # Page 3: Equity Curve
            if not equity_curve.empty:
                fig = self.perf_plotter.plot_equity_curve(
                    equity_curve, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 4: Returns Distribution
            if not returns.empty:
                fig = self.perf_plotter.plot_returns_distribution(
                    returns, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 5: Rolling Metrics
            if not returns.empty:
                fig = self.perf_plotter.plot_rolling_metrics(
                    returns, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 6: Monthly Heatmap
            if not returns.empty:
                fig = self.perf_plotter.plot_monthly_heatmap(
                    returns, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 7: Trade Analysis
            if not trades.empty:
                fig = self.perf_plotter.plot_trade_analysis(
                    trades, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 8: Feature Importance
            if not feature_importance.empty:
                fig = self.factor_plotter.plot_feature_importance(
                    feature_importance, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 9: VaR Analysis
            if not returns.empty:
                fig = self.risk_plotter.plot_var_analysis(
                    returns, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 10: Risk Dashboard
            if not returns.empty:
                fig = self.risk_plotter.plot_risk_dashboard(
                    returns, equity_curve, show=False
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 11: Trade Details Table
            if not trades.empty:
                self._create_trades_table_page(pdf, trades)
        
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
        ax.axhline(y=0.83, xmin=0.1, xmax=0.9, color='#2E86AB', linewidth=2)
        
        # Timestamp
        fig.text(0.5, 0.81, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                fontsize=11, ha='center', va='top', color='#888888')
        
        # Configuration Box
        config_text = f"""
┌─────────────────────────────────────────────────────────────┐
│                      CONFIGURATION                          │
├─────────────────────────────────────────────────────────────┤
│  Period:           {config.get('start_date', 'N/A'):>15} → {config.get('end_date', 'N/A'):<15} │
│  Universe:         {str(config.get('universe', 'N/A')):>40} │
│  Initial Capital:  ₹{config.get('initial_capital', 0):>38,.0f} │
│  Strategy:         {str(config.get('strategy', 'ML Multi-Factor')):>40} │
└─────────────────────────────────────────────────────────────┘
"""
        fig.text(0.5, 0.75, config_text, fontsize=10, ha='center', va='top',
                fontfamily='monospace', color='#333333')
        
        # Key Metrics Summary
        cagr = metrics.get('cagr', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        total_return = metrics.get('total_return', 0) * 100
        
        metrics_text = f"""
╔═════════════════════════════════════════════════════════════╗
║                    EXECUTIVE SUMMARY                        ║
╠═════════════════════════════════════════════════════════════╣
║                                                             ║
║   RETURNS                          RISK-ADJUSTED            ║
║   ───────────────────────          ─────────────────────    ║
║   Total Return:    {total_return:>8.2f}%         Sharpe Ratio:  {sharpe:>8.2f}   ║
║   CAGR:            {cagr:>8.2f}%         Sortino Ratio: {metrics.get('sortino_ratio', 0):>8.2f}   ║
║                                    Calmar Ratio:  {metrics.get('calmar_ratio', 0):>8.2f}   ║
║                                                             ║
║   RISK                             TRADING                  ║
║   ───────────────────────          ─────────────────────    ║
║   Max Drawdown:    {max_dd:>8.2f}%         Win Rate:      {win_rate:>8.1f}%  ║
║   Volatility:      {metrics.get('volatility', 0)*100:>8.2f}%         Profit Factor: {metrics.get('profit_factor', 0):>8.2f}   ║
║   VaR (95%):       {metrics.get('var_95', 0)*100:>8.2f}%         Total Trades:  {metrics.get('total_trades', 0):>8.0f}   ║
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
            ('CAGR', f"{metrics.get('cagr', 0)*100:.2f}%", '#2E86AB'),
            ('Total Return', f"{metrics.get('total_return', 0)*100:.2f}%", '#2E86AB'),
            ('Avg Daily', f"{metrics.get('avg_daily_return', 0)*100:.4f}%", '#2E86AB'),
            ('Best Day', f"{metrics.get('best_day', 0)*100:.2f}%", '#28A745'),
            
            # Row 2: Risk Metrics
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0)*100:.2f}%", '#DC3545'),
            ('Volatility', f"{metrics.get('volatility', 0)*100:.2f}%", '#FFC107'),
            ('VaR (95%)', f"{metrics.get('var_95', 0)*100:.2f}%", '#DC3545'),
            ('Worst Day', f"{metrics.get('worst_day', 0)*100:.2f}%", '#DC3545'),
            
            # Row 3: Risk-Adjusted
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}", '#2E86AB'),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}", '#2E86AB'),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}", '#2E86AB'),
            ('Info Ratio', f"{metrics.get('information_ratio', 0):.2f}", '#2E86AB'),
            
            # Row 4: Trading
            ('Win Rate', f"{metrics.get('win_rate', 0)*100:.1f}%", '#28A745'),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}", '#28A745'),
            ('Total Trades', f"{metrics.get('total_trades', 0):.0f}", '#6C757D'),
            ('Avg Trade', f"{metrics.get('avg_trade_return', 0)*100:.3f}%", '#6C757D'),
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
        fig.text(0.5, 0.95, 'Recent Trades',
                fontsize=16, fontweight='bold', ha='center', color='#2E86AB')
        
        # Prepare data
        recent_trades = trades.tail(n_rows).copy()
        
        if 'date' in recent_trades.columns:
            recent_trades = recent_trades.sort_values('date', ascending=False)
        
        # Select columns
        display_cols = []
        col_widths = []
        
        if 'date' in recent_trades.columns:
            display_cols.append('date')
            col_widths.append(0.15)
        if 'ticker' in recent_trades.columns:
            display_cols.append('ticker')
            col_widths.append(0.12)
        if 'side' in recent_trades.columns:
            display_cols.append('side')
            col_widths.append(0.08)
        if 'pnl' in recent_trades.columns:
            display_cols.append('pnl')
            col_widths.append(0.15)
        if 'return_pct' in recent_trades.columns:
            display_cols.append('return_pct')
            col_widths.append(0.12)
        
        if display_cols:
            table_data = recent_trades[display_cols].values.tolist()
            
            # Format data
            formatted_data = []
            for row in table_data:
                formatted_row = []
                for i, val in enumerate(row):
                    col = display_cols[i]
                    if col == 'pnl':
                        formatted_row.append(f"₹{val:,.0f}")
                    elif col == 'return_pct':
                        formatted_row.append(f"{val*100:.2f}%")
                    elif col == 'date':
                        formatted_row.append(str(val)[:10])
                    else:
                        formatted_row.append(str(val))
                formatted_data.append(formatted_row)
            
            # Create table
            table = ax.table(
                cellText=formatted_data,
                colLabels=[c.upper() for c in display_cols],
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
                    if row[pnl_col] > 0:
                        table[(i, pnl_col)].set_text_props(color='#28A745')
                    else:
                        table[(i, pnl_col)].set_text_props(color='#DC3545')
        
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
        metrics = backtest_results.get('metrics', {})
        config = backtest_results.get('config', {})
        trades = backtest_results.get('trades', pd.DataFrame())
        
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
        
        config_table.add_row("Period", f"{config.get('start_date', 'N/A')} → {config.get('end_date', 'N/A')}")
        config_table.add_row("Universe", str(config.get('universe', 'N/A')))
        config_table.add_row("Initial Capital", f"₹{config.get('initial_capital', 0):,.0f}")
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
        
        cagr = metrics.get('cagr', 0)
        perf_table.add_row(
            "CAGR",
            f"{cagr*100:.2f}%",
            get_status(cagr, 0.12, 0)
        )
        
        total_ret = metrics.get('total_return', 0)
        perf_table.add_row(
            "Total Return",
            f"{total_ret*100:.2f}%",
            get_status(total_ret, 0.20, 0)
        )
        
        sharpe = metrics.get('sharpe_ratio', 0)
        perf_table.add_row(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            get_status(sharpe, 1.0, 0.5)
        )
        
        sortino = metrics.get('sortino_ratio', 0)
        perf_table.add_row(
            "Sortino Ratio",
            f"{sortino:.2f}",
            get_status(sortino, 1.5, 0.5)
        )
        
        max_dd = metrics.get('max_drawdown', 0)
        perf_table.add_row(
            "Max Drawdown",
            f"{max_dd*100:.2f}%",
            get_status(abs(max_dd), 0.10, 0.25, higher_is_better=False)
        )
        
        vol = metrics.get('volatility', 0)
        perf_table.add_row(
            "Volatility (Ann.)",
            f"{vol*100:.2f}%",
            get_status(vol, 0.15, 0.30, higher_is_better=False)
        )
        
        calmar = metrics.get('calmar_ratio', 0)
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
        
        win_rate = metrics.get('win_rate', 0)
        trade_table.add_row("Win Rate", f"{win_rate*100:.1f}%")
        trade_table.add_row("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        trade_table.add_row("Total Trades", f"{metrics.get('total_trades', 0):,.0f}")
        
        if not trades.empty and 'pnl' in trades.columns:
            trade_table.add_row("Total P&L", f"₹{trades['pnl'].sum():,.0f}")
            trade_table.add_row("Avg P&L/Trade", f"₹{trades['pnl'].mean():,.0f}")
            trade_table.add_row("Best Trade", f"₹{trades['pnl'].max():,.0f}")
            trade_table.add_row("Worst Trade", f"₹{trades['pnl'].min():,.0f}")
        
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
        
        risk_table.add_row("VaR (95%)", f"{metrics.get('var_95', 0)*100:.2f}%")
        risk_table.add_row("CVaR (95%)", f"{metrics.get('cvar_95', 0)*100:.2f}%")
        risk_table.add_row("Skewness", f"{metrics.get('skewness', 0):.4f}")
        risk_table.add_row("Kurtosis", f"{metrics.get('kurtosis', 0):.4f}")
        risk_table.add_row("Beta", f"{metrics.get('beta', 0):.2f}")
        risk_table.add_row("Alpha (Ann.)", f"{metrics.get('alpha', 0)*100:.2f}%")
        
        console.print(risk_table)
        console.print()
        
        # Overall Assessment Panel
        cagr_val = metrics.get('cagr', 0)
        sharpe_val = metrics.get('sharpe_ratio', 0)
        
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
        
        print(f"\n📅 Period: {config.get('start_date', 'N/A')} → {config.get('end_date', 'N/A')}")
        print(f"🏛️ Universe: {config.get('universe', 'N/A')}")
        print(f"💰 Initial Capital: ₹{config.get('initial_capital', 0):,.0f}")
        print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "-" * 70)
        print(" PERFORMANCE METRICS")
        print("-" * 70)
        
        print(f"\n  {'RETURNS':<25} {'RISK-ADJUSTED':<25}")
        print(f"  {'-'*23}   {'-'*23}")
        print(f"  CAGR:           {metrics.get('cagr', 0)*100:>8.2f}%    Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Total Return:   {metrics.get('total_return', 0)*100:>8.2f}%    Sortino Ratio: {metrics.get('sortino_ratio', 0):>8.2f}")
        print(f"  Volatility:     {metrics.get('volatility', 0)*100:>8.2f}%    Calmar Ratio:  {metrics.get('calmar_ratio', 0):>8.2f}")
        
        print(f"\n  {'RISK':<25} {'TRADING':<25}")
        print(f"  {'-'*23}   {'-'*23}")
        print(f"  Max Drawdown:   {metrics.get('max_drawdown', 0)*100:>8.2f}%    Win Rate:      {metrics.get('win_rate', 0)*100:>8.1f}%")
        print(f"  VaR (95%):      {metrics.get('var_95', 0)*100:>8.2f}%    Profit Factor: {metrics.get('profit_factor', 0):>8.2f}")
        print(f"  CVaR (95%):     {metrics.get('cvar_95', 0)*100:>8.2f}%    Total Trades:  {metrics.get('total_trades', 0):>8.0f}")
        
        if not trades.empty and 'pnl' in trades.columns:
            print(f"\n  {'TRADE DETAILS':<25}")
            print(f"  {'-'*23}")
            print(f"  Total P&L:      ₹{trades['pnl'].sum():>12,.0f}")
            print(f"  Avg P&L/Trade:  ₹{trades['pnl'].mean():>12,.0f}")
            print(f"  Best Trade:     ₹{trades['pnl'].max():>12,.0f}")
            print(f"  Worst Trade:    ₹{trades['pnl'].min():>12,.0f}")
        
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
            'config': backtest_results.get('config', {}),
            'metrics': backtest_results.get('metrics', {}),
            'summary': {}
        }
        
        # Add summary data
        equity = backtest_results.get('equity_curve', pd.Series())
        if not equity.empty:
            export_data['summary']['start_date'] = str(equity.index[0])
            export_data['summary']['end_date'] = str(equity.index[-1])
            export_data['summary']['start_value'] = float(equity.iloc[0])
            export_data['summary']['end_value'] = float(equity.iloc[-1])
        
        trades = backtest_results.get('trades', pd.DataFrame())
        if not trades.empty:
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
            print("❌ openpyxl not installed. Run: pip install openpyxl")
            return ""
        
        if report_name is None:
            report_name = f"backtest_{self.timestamp}"
        
        excel_path = self.output_dir / f"{report_name}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Sheet 1: Metrics
            metrics = backtest_results.get('metrics', {})
            if metrics:
                metrics_df = pd.DataFrame([
                    {'Metric': k, 'Value': v} for k, v in metrics.items()
                ])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Sheet 2: Configuration
            config = backtest_results.get('config', {})
            if config:
                config_df = pd.DataFrame([
                    {'Parameter': k, 'Value': str(v)} for k, v in config.items()
                ])
                config_df.to_excel(writer, sheet_name='Config', index=False)
            
            # Sheet 3: Trades
            trades = backtest_results.get('trades', pd.DataFrame())
            if not trades.empty:
                trades.to_excel(writer, sheet_name='Trades', index=False)
            
            # Sheet 4: Returns
            returns = backtest_results.get('returns', pd.Series())
            if not returns.empty:
                returns_df = returns.to_frame('daily_return')
                returns_df['cumulative_return'] = (1 + returns).cumprod() - 1
                returns_df.to_excel(writer, sheet_name='Returns')
            
            # Sheet 5: Equity Curve
            equity = backtest_results.get('equity_curve', pd.Series())
            if not equity.empty:
                equity_df = equity.to_frame('portfolio_value')
                equity_df.to_excel(writer, sheet_name='Equity')
            
            # Sheet 6: Feature Importance
            feature_imp = backtest_results.get('feature_importance', pd.DataFrame())
            if not feature_imp.empty:
                feature_imp.to_excel(writer, sheet_name='Features', index=False)
        
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
        metrics = backtest_results.get('metrics', {})
        if metrics:
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            metrics_df.index.name = 'Metric'
            metrics_df.to_csv(csv_dir / 'metrics.csv')
        
        # Trades
        trades = backtest_results.get('trades', pd.DataFrame())
        if not trades.empty:
            trades.to_csv(csv_dir / 'trades.csv', index=False)
        
        # Returns
        returns = backtest_results.get('returns', pd.Series())
        if not returns.empty:
            returns.to_frame('daily_return').to_csv(csv_dir / 'returns.csv')
        
        # Equity
        equity = backtest_results.get('equity_curve', pd.Series())
        if not equity.empty:
            equity.to_frame('portfolio_value').to_csv(csv_dir / 'equity.csv')
        
        # Features
        feature_imp = backtest_results.get('feature_importance', pd.DataFrame())
        if not feature_imp.empty:
            feature_imp.to_csv(csv_dir / 'features.csv', index=False)
        
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
        
        equity = backtest_results.get('equity_curve', pd.Series())
        returns = backtest_results.get('returns', pd.Series())
        trades = backtest_results.get('trades', pd.DataFrame())
        feature_imp = backtest_results.get('feature_importance', pd.DataFrame())
        
        saved = []
        
        # Generate charts
        if not equity.empty:
            try:
                self.perf_plotter.plot_equity_curve(
                    equity, save_path=str(charts_dir / 'equity_curve.png'), show=False
                )
                saved.append('equity_curve.png')
            except Exception as e:
                print(f"⚠️ equity_curve failed: {e}")
        
        if not returns.empty:
            try:
                self.perf_plotter.plot_returns_distribution(
                    returns, save_path=str(charts_dir / 'returns_distribution.png'), show=False
                )
                saved.append('returns_distribution.png')
            except Exception as e:
                print(f"⚠️ returns_distribution failed: {e}")
            
            try:
                self.perf_plotter.plot_rolling_metrics(
                    returns, save_path=str(charts_dir / 'rolling_metrics.png'), show=False
                )
                saved.append('rolling_metrics.png')
            except Exception as e:
                print(f"⚠️ rolling_metrics failed: {e}")
            
            try:
                self.perf_plotter.plot_monthly_heatmap(
                    returns, save_path=str(charts_dir / 'monthly_heatmap.png'), show=False
                )
                saved.append('monthly_heatmap.png')
            except Exception as e:
                print(f"⚠️ monthly_heatmap failed: {e}")
            
            try:
                self.risk_plotter.plot_var_analysis(
                    returns, save_path=str(charts_dir / 'var_analysis.png'), show=False
                )
                saved.append('var_analysis.png')
            except Exception as e:
                print(f"⚠️ var_analysis failed: {e}")
            
            try:
                self.risk_plotter.plot_risk_dashboard(
                    returns, equity, save_path=str(charts_dir / 'risk_dashboard.png'), show=False
                )
                saved.append('risk_dashboard.png')
            except Exception as e:
                print(f"⚠️ risk_dashboard failed: {e}")
        
        if not trades.empty:
            try:
                self.perf_plotter.plot_trade_analysis(
                    trades, save_path=str(charts_dir / 'trade_analysis.png'), show=False
                )
                saved.append('trade_analysis.png')
            except Exception as e:
                print(f"⚠️ trade_analysis failed: {e}")
        
        if not feature_imp.empty:
            try:
                self.factor_plotter.plot_feature_importance(
                    feature_imp, save_path=str(charts_dir / 'feature_importance.png'), show=False
                )
                saved.append('feature_importance.png')
            except Exception as e:
                print(f"⚠️ feature_importance failed: {e}")
        
        print(f"✅ {len(saved)} charts saved: {charts_dir}")
        return str(charts_dir)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_report(
    backtest_results: Dict[str, Any],
    output_dir: str = "reports",
    formats: List[str] = ['pdf', 'terminal', 'json'],
    report_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Quick function to generate reports.
    
    Args:
        backtest_results: Backtest results dictionary
        output_dir: Output directory
        formats: List of formats ('pdf', 'terminal', 'json', 'excel', 'csv', 'png', 'pickle')
        report_name: Base name for files
        
    Returns:
        Dictionary mapping format to file path
    
    Example:
        >>> outputs = generate_report(results, formats=['pdf', 'json'])
    """
    generator = ReportGenerator(output_dir)
    return generator.generate_full_report(backtest_results, formats, report_name)


def print_metrics(
    metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None
):
    """
    Quick function to print metrics to terminal.
    
    Args:
        metrics: Metrics dictionary
        config: Optional config dictionary
    
    Example:
        >>> print_metrics({'cagr': 0.15, 'sharpe_ratio': 1.5})
    """
    generator = ReportGenerator()
    generator.print_terminal_report({
        'metrics': metrics,
        'config': config or {},
        'trades': pd.DataFrame()
    })


def save_all_charts(
    backtest_results: Dict[str, Any],
    output_dir: str = "reports/charts"
) -> str:
    """
    Quick function to save all charts.
    
    Args:
        backtest_results: Backtest results dictionary
        output_dir: Output directory
        
    Returns:
        Path to charts directory
    
    Example:
        >>> save_all_charts(results, 'my_charts')
    """
    generator = ReportGenerator(output_dir)
    return generator.save_all_charts(backtest_results)


def export_to_json(
    backtest_results: Dict[str, Any],
    output_path: str = "reports/metrics.json"
) -> str:
    """
    Quick function to export to JSON.
    
    Args:
        backtest_results: Backtest results dictionary
        output_path: Output file path
        
    Returns:
        Path to JSON file
    """
    output_dir = str(Path(output_path).parent)
    filename = Path(output_path).stem
    
    generator = ReportGenerator(output_dir)
    return generator.export_to_json(backtest_results, filename)


def export_to_excel(
    backtest_results: Dict[str, Any],
    output_path: str = "reports/backtest.xlsx"
) -> str:
    """
    Quick function to export to Excel.
    
    Args:
        backtest_results: Backtest results dictionary
        output_path: Output file path
        
    Returns:
        Path to Excel file
    """
    output_dir = str(Path(output_path).parent)
    filename = Path(output_path).stem
    
    generator = ReportGenerator(output_dir)
    return generator.export_to_excel(backtest_results, filename)


def export_to_csv(
    backtest_results: Dict[str, Any],
    output_dir: str = "reports/csv"
) -> str:
    """
    Quick function to export to CSV.
    
    Args:
        backtest_results: Backtest results dictionary
        output_dir: Output directory
        
    Returns:
        Path to CSV directory
    """
    generator = ReportGenerator(output_dir)
    return generator.export_to_csv(backtest_results)