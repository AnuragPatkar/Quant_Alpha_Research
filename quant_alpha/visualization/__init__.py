"""
Visualization Module
====================
Professional charts, reports, and dashboards for quant research.

Features:
- Performance visualization (equity, drawdown, returns)
- Factor analysis plots (importance, SHAP, correlations)
- Risk visualization (VaR, drawdown analysis)
- Report generation (PDF, Excel, JSON)
- Interactive dashboards (optional: requires streamlit)

Author: [Your Name]
Last Updated: 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# ===========================================
# Global Plot Configuration
# ===========================================

PLOT_CONFIG = {
    'figure_size': (12, 8),
    'figure_size_small': (8, 5),
    'figure_size_wide': (14, 6),
    'figure_size_tall': (10, 12),
    'dpi': 100,
    'save_dpi': 300,
    'font_size': 11,
    'title_size': 14,
    'label_size': 11,
    'tick_size': 10,
    'legend_size': 10,
    'line_width': 2,
    'grid_alpha': 0.3,
    'grid_style': '--',
    'colors': {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'positive': '#28A745',
        'negative': '#DC3545',
        'warning': '#FFC107',
        'info': '#17A2B8',
        'neutral': '#6C757D',
        'benchmark': '#FF9800',
        'strategy': '#2E86AB',
        'background': '#FFFFFF',
        'text': '#212529',
        'grid': '#E0E0E0'
    },
    'palettes': {
        'default': ['#2E86AB', '#A23B72', '#28A745', '#DC3545', '#FFC107'],
        'sequential': 'Blues',
        'diverging': 'RdYlGn',
    }
}

# ===========================================
# Setup Functions
# ===========================================

def setup_plot_style(style: str = 'default', dark_mode: bool = False) -> None:
    """
    Setup matplotlib style globally.
    
    Args:
        style: Style name ('default', 'minimal', 'publication')
        dark_mode: Whether to use dark background
    """
    # Try multiple style options for compatibility
    style_options = [
        'seaborn-v0_8-whitegrid',
        'seaborn-whitegrid', 
        'ggplot',
        'default'
    ]
    
    for s in style_options:
        try:
            plt.style.use(s)
            break
        except OSError:
            continue
    
    # Set seaborn palette
    sns.set_palette("husl")
    
    # Apply custom settings
    plt.rcParams.update({
        'figure.figsize': PLOT_CONFIG['figure_size'],
        'figure.dpi': PLOT_CONFIG['dpi'],
        'savefig.dpi': PLOT_CONFIG['save_dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': PLOT_CONFIG['font_size'],
        'axes.titlesize': PLOT_CONFIG['title_size'],
        'axes.labelsize': PLOT_CONFIG['label_size'],
        'xtick.labelsize': PLOT_CONFIG['tick_size'],
        'ytick.labelsize': PLOT_CONFIG['tick_size'],
        'legend.fontsize': PLOT_CONFIG['legend_size'],
        'lines.linewidth': PLOT_CONFIG['line_width'],
        'grid.alpha': PLOT_CONFIG['grid_alpha'],
        'grid.linestyle': PLOT_CONFIG['grid_style'],
        'figure.autolayout': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
    })
    
    if dark_mode:
        plt.rcParams.update({
            'figure.facecolor': '#1A1A2E',
            'axes.facecolor': '#1A1A2E',
            'axes.edgecolor': '#FFFFFF',
            'axes.labelcolor': '#FFFFFF',
            'text.color': '#FFFFFF',
            'xtick.color': '#FFFFFF',
            'ytick.color': '#FFFFFF',
            'grid.color': '#333333',
        })
    
    logger.debug(f"Plot style set to: {style}, dark_mode={dark_mode}")


def get_color(name: str) -> str:
    """Get color by name from config."""
    return PLOT_CONFIG['colors'].get(name, PLOT_CONFIG['colors']['primary'])


def get_colors(n: int = 5) -> list:
    """Get n colors from default palette."""
    palette = PLOT_CONFIG['palettes']['default']
    if n <= len(palette):
        return palette[:n]
    return sns.color_palette("husl", n).as_hex()


# Apply default style on import
setup_plot_style('default')

# ===========================================
# Core Imports
# ===========================================

try:
    from .plots import (
        PerformancePlotter,
        FactorPlotter,
        RiskPlotter,
        quick_plot_all,
        plot_equity_curve,
        plot_drawdown,
        plot_returns_distribution,
        plot_ic_series,
        plot_feature_importance,
        plot_quantile_returns,
    )
    PLOTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import plots module: {e}")
    PLOTS_AVAILABLE = False
    
    # Placeholder
    class PerformancePlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("plots module not available")
    
    FactorPlotter = PerformancePlotter
    RiskPlotter = PerformancePlotter
    quick_plot_all = None

try:
    from .reports import (
        ReportGenerator,
        generate_report,
        print_metrics,
        save_all_charts,
        export_to_json,
        export_to_excel,
        export_to_csv,
    )
    REPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import reports module: {e}")
    REPORTS_AVAILABLE = False
    
    class ReportGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("reports module not available")
    
    generate_report = None
    print_metrics = None

# ===========================================
# Optional Dashboard Imports
# ===========================================

DASHBOARD_AVAILABLE = False

try:
    import streamlit
    import plotly
    DASHBOARD_AVAILABLE = True
    
    from .dashboards import (
        QuantDashboard,
        DashboardConfig,
        run_dashboard,
    )
except ImportError:
    logger.debug("Dashboard dependencies not installed (streamlit, plotly)")
    
    class QuantDashboard:
        """Placeholder when streamlit/plotly not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Dashboard requires streamlit and plotly.\n"
                "Install with: pip install streamlit plotly"
            )
    
    class DashboardConfig:
        """Placeholder config class."""
        pass
    
    def run_dashboard(*args, **kwargs):
        """Placeholder function."""
        print("❌ Dashboard not available!")
        print("   Install with: pip install streamlit plotly")
        return None

# ===========================================
# Public API
# ===========================================

__all__ = [
    # Configuration
    'PLOT_CONFIG',
    'setup_plot_style',
    'get_color',
    'get_colors',
    
    # Availability flags
    'PLOTS_AVAILABLE',
    'REPORTS_AVAILABLE', 
    'DASHBOARD_AVAILABLE',
    
    # Plotters
    'PerformancePlotter',
    'FactorPlotter',
    'RiskPlotter',
    'quick_plot_all',
    
    # Convenience functions (if available)
    'plot_equity_curve',
    'plot_drawdown',
    'plot_returns_distribution',
    'plot_ic_series',
    'plot_feature_importance',
    'plot_quantile_returns',
    
    # Reports
    'ReportGenerator',
    'generate_report',
    'print_metrics',
    'save_all_charts',
    'export_to_json',
    'export_to_excel',
    'export_to_csv',
    
    # Dashboards (optional)
    'QuantDashboard',
    'DashboardConfig',
    'run_dashboard',
    
    # Module info
    'info',
]

__version__ = "2.0.0"

# ===========================================
# Module Info
# ===========================================

def info() -> None:
    """Print module information and availability status."""
    
    plots_status = "✅ Available" if PLOTS_AVAILABLE else "❌ Not available"
    reports_status = "✅ Available" if REPORTS_AVAILABLE else "❌ Not available"
    dashboard_status = "✅ Available" if DASHBOARD_AVAILABLE else "❌ Not installed"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            QUANT ALPHA - VISUALIZATION MODULE                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Plotters:           {plots_status:<30}  ║
║     • PerformancePlotter - Equity, returns, drawdown         ║
║     • FactorPlotter - Feature importance, IC, SHAP           ║
║     • RiskPlotter - VaR, CVaR, risk analysis                 ║
║                                                              ║
║  📄 Reports:            {reports_status:<30}  ║
║     • PDF/PNG chart generation                               ║
║     • JSON/Excel/CSV data export                             ║
║     • Terminal-friendly metrics display                      ║
║                                                              ║
║  🖥️  Dashboard:          {dashboard_status:<30}  ║
║     • Interactive Streamlit dashboard                        ║
║     • Install: pip install streamlit plotly                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


# Auto-show info in interactive mode (optional)
# Uncomment if you want info() to run on import in notebooks
# if hasattr(__builtins__, '__IPYTHON__'):
#     info()