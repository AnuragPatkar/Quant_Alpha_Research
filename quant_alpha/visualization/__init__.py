"""
Visualization Module - Pure Python
===================================
Professional charts, reports, and dashboards.
100% Python - No HTML strings!
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ===========================================
# Global Plot Configuration
# ===========================================

PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 150,
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
        'background': '#FFFFFF',
        'text': '#212529'
    },
    'dark_colors': {
        'primary': '#00D4AA',
        'secondary': '#FF6B6B',
        'positive': '#00D4AA',
        'negative': '#FF6B6B',
        'warning': '#FFD93D',
        'info': '#6BCB77',
        'neutral': '#888888',
        'benchmark': '#FFA500',
        'background': '#1A1A2E',
        'text': '#FFFFFF'
    }
}

# ===========================================
# Apply Global Matplotlib Settings
# ===========================================

def setup_plot_style(style: str = 'default'):
    """Setup matplotlib style globally."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'figure.figsize': PLOT_CONFIG['figure_size'],
        'figure.dpi': PLOT_CONFIG['dpi'],
        'savefig.dpi': PLOT_CONFIG['save_dpi'],
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
        'axes.spines.right': False
    })

# Apply default style
setup_plot_style('default')

# ===========================================
# Core Imports (Always Available)
# ===========================================

from .plots import (
    PerformancePlotter,
    FactorPlotter,
    RiskPlotter,
    quick_plot_all
)

from .reports import (
    ReportGenerator,
    generate_report,
    print_metrics,
    save_all_charts,
    export_to_json,
    export_to_excel,
    export_to_csv
)

# ===========================================
# Optional Imports (Dashboard - needs streamlit/plotly)
# ===========================================

# Check if streamlit and plotly are available
DASHBOARD_AVAILABLE = False

try:
    import streamlit
    import plotly
    DASHBOARD_AVAILABLE = True
except ImportError:
    pass

if DASHBOARD_AVAILABLE:
    from .dashboards import (
        QuantDashboard,
        DashboardConfig,
        run_dashboard
    )
else:
    # Placeholder classes when dashboard not available
    class QuantDashboard:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Dashboard requires streamlit and plotly. "
                "Install with: pip install streamlit plotly"
            )
    
    class DashboardConfig:
        pass
    
    def run_dashboard():
        print("❌ Dashboard not available!")
        print("   Install with: pip install streamlit plotly")

# ===========================================
# Public API
# ===========================================

__all__ = [
    # Configuration
    'PLOT_CONFIG',
    'setup_plot_style',
    'DASHBOARD_AVAILABLE',
    
    # Plotters (always available)
    'PerformancePlotter',
    'FactorPlotter',
    'RiskPlotter',
    'quick_plot_all',
    
    # Reports (always available)
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
    'run_dashboard'
]

__version__ = "1.0.0"

# ===========================================
# Info Function
# ===========================================

def info():
    """Print module information."""
    dashboard_status = "✅ Available" if DASHBOARD_AVAILABLE else "❌ Not installed (pip install streamlit plotly)"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           QUANT ALPHA - VISUALIZATION MODULE             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📊 Plotters (matplotlib):                               ║
║     • PerformancePlotter - Equity, returns, drawdown     ║
║     • FactorPlotter - Feature importance, SHAP           ║
║     • RiskPlotter - VaR, CVaR, risk metrics              ║
║                                                          ║
║  📄 Reports:                                             ║
║     • PDF reports (matplotlib)                           ║
║     • Terminal output (Rich - optional)                  ║
║     • JSON/Excel/CSV exports                             ║
║                                                          ║
║  🖥️ Dashboard: {dashboard_status:<30} ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)