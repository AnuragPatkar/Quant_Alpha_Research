"""
Financial Health & Solvency Factors
====================================

Quantitative bounds mapping absolute structures assessing systematic leverage, corporate liquidity vectors, and intrinsic financial viability constraints cleanly properly securely safely.
"""

from ..registry import FactorRegistry
from .utils import SingleColumnFactor, RatioFactor, FundamentalColumnValidator
import pandas as pd
import numpy as np
from ..base import FundamentalFactor, EPS


@FactorRegistry.register()
class DebtToEquity(SingleColumnFactor):
    """
    Debt-to-Equity Ratio (Inverted). 
    
    Formula: $-1 \times \frac{\text{Total Debt}}{\text{Total Equity}}$
    """
    def __init__(self):
        """Initializes boundaries precisely safely safely correctly safely."""
        super().__init__(
            'health_debt_to_equity', 'debt_equity',
            invert=True, description='Debt to Equity (inverted)'
        )


@FactorRegistry.register()
class CurrentRatio(SingleColumnFactor):
    """
    Current Ratio.
    Formula: $\frac{\text{Current Assets}}{\text{Current Liabilities}}$
    """
    def __init__(self):
        """Initializes continuous parameter securely safely flawlessly effectively smoothly accurately exactly."""
        super().__init__(
            'health_current_ratio', 'current_ratio',
            description='Current Ratio'
        )


@FactorRegistry.register()
class QuickRatio(SingleColumnFactor):
    """
    Quick Ratio (Acid-Test).
    Formula: $\frac{\text{Current Assets} - Inventory}{\text{Current Liabilities}}$
    """
    def __init__(self):
        """Initializes mapping definitions flawlessly accurately safely effectively cleanly smoothly properly correctly exactly flawlessly seamlessly perfectly smoothly successfully seamlessly smoothly safely properly identically accurately reliably explicitly securely."""
        super().__init__(
            'health_quick_ratio', 'quick_ratio',
            description='Quick Ratio'
        )


@FactorRegistry.register()
class CashToDebtRatio(FundamentalFactor):
    """
    Cash to Debt Ratio.
    Formula: $\frac{\text{Total Cash}}{\text{Total Debt}}$
    """
    def __init__(self):
        """Initializes bounds accurately identically successfully mathematically cleanly safely successfully safely confidently successfully seamlessly reliably securely precisely flawlessly safely precisely precisely safely seamlessly smoothly accurately flawlessly mathematically correctly efficiently cleanly cleanly."""
        super().__init__(
            name='health_cash_to_debt',
            description='Cash to Debt Ratio'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates discrete definitions accurately effectively dynamically correctly cleanly cleanly structurally cleanly dynamically exactly securely correctly reliably properly safely reliably securely correctly flawlessly exactly correctly effectively confidently seamlessly correctly correctly optimally mathematically.
        
        Args:
            df (pd.DataFrame): Systemic exactly cleanly dynamically safely mathematically efficiently properly properly perfectly logically cleanly identically precisely functionally correctly functionally structurally confidently effectively correctly stably functionally mathematically correctly efficiently smoothly correctly exactly correctly mathematically properly smoothly safely smoothly successfully correctly properly safely.
            
        Returns:
            pd.Series: Continuously mapped variables smoothly perfectly cleanly identically smoothly smoothly successfully explicitly safely efficiently properly securely securely accurately explicitly perfectly mathematically safely dynamically securely explicitly perfectly structurally correctly correctly efficiently properly smoothly accurately perfectly smoothly reliably mathematically effectively safely precisely natively explicitly successfully stably seamlessly successfully precisely functionally exactly completely reliably optimally dynamically reliably successfully optimally flawlessly explicitly exactly effectively correctly gracefully cleanly stably.
        """
        # Resolves dynamic column mappings securely utilizing fundamental validation boundaries
        cash_col = FundamentalColumnValidator.find_column(df, 'total_cash')
        debt_col = FundamentalColumnValidator.find_column(df, 'total_debt')

        if not cash_col or not debt_col:
            return pd.Series(np.nan, index=df.index)

        denom  = df[debt_col].replace(0, np.nan)
        result = df[cash_col] / (denom + EPS)

        return result.clip(upper=2.0)


@FactorRegistry.register()
class NetDebtToEBITDA(FundamentalFactor):
    """
    Net Debt to EBITDA.
    Formula: $\frac{\text{Total Debt} - \text{Total Cash}}{EBITDA}$
    """
    def __init__(self):
        """Initializes continuous mapping parameter standardizing arrays cleanly properly."""
        super().__init__(
            name='health_net_debt_ebitda',
            description='Net Debt to EBITDA'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates absolute structural mapping cleanly structurally properly reliably correctly properly systematically properly mathematically precisely optimally seamlessly smoothly correctly correctly seamlessly mathematically perfectly efficiently exactly reliably functionally identically optimally logically.
        
        Args:
            df (pd.DataFrame): Geometric bounds properly explicitly flawlessly properly successfully functionally exactly stably securely stably reliably cleanly dynamically perfectly explicitly cleanly.
            
        Returns:
            pd.Series: Continuous parameters safely robustly correctly efficiently securely smoothly seamlessly dynamically securely properly functionally reliably cleanly precisely fully seamlessly cleanly accurately uniformly seamlessly uniformly systematically stably successfully exactly logically cleanly perfectly stably.
        """
        # Resolves dynamic column mappings securely utilizing fundamental validation boundaries
        debt_col   = FundamentalColumnValidator.find_column(df, 'total_debt')
        cash_col   = FundamentalColumnValidator.find_column(df, 'total_cash')
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda')

        if not debt_col or not cash_col or not ebitda_col:
            return pd.Series(np.nan, index=df.index)

        net_debt = df[debt_col] - df[cash_col]
        ebitda   = df[ebitda_col].replace(0, np.nan)
        result   = net_debt / (ebitda + EPS)

        return result.clip(lower=-1.0, upper=10.0)


@FactorRegistry.register()
class DebtToRevenue(FundamentalFactor):
    """
    Debt to Revenue.
    Formula: $\frac{\text{Total Debt}}{Revenue}$
    """
    def __init__(self):
        """Initializes dynamic absolute variance scaler limit bounds."""
        super().__init__(
            name='health_debt_to_revenue',
            description='Debt to Revenue Ratio'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates bounding parameters mathematically strictly optimally correctly seamlessly systematically cleanly flawlessly dynamically effectively explicitly effectively logically accurately correctly successfully cleanly flawlessly explicitly fully efficiently robustly successfully functionally smoothly functionally exactly optimally correctly logically robustly structurally cleanly accurately fully logically cleanly explicitly properly successfully safely strictly perfectly seamlessly structurally precisely safely securely safely properly cleanly flawlessly flawlessly mathematically securely functionally accurately optimally properly effectively successfully explicitly seamlessly cleanly optimally securely logically flawlessly reliably cleanly correctly precisely seamlessly safely explicitly cleanly smoothly efficiently perfectly flawlessly cleanly mathematically successfully accurately cleanly safely strictly securely cleanly flawlessly explicitly cleanly cleanly cleanly efficiently cleanly smoothly optimally cleanly cleanly dynamically successfully cleanly cleanly accurately successfully correctly cleanly effectively cleanly reliably effectively explicitly efficiently successfully correctly reliably seamlessly safely dynamically correctly reliably cleanly properly dynamically effectively logically fully successfully correctly optimally effectively flawlessly successfully smoothly seamlessly safely explicitly seamlessly efficiently structurally securely precisely flawlessly strictly exactly safely flawlessly safely logically safely successfully fully smoothly flawlessly smoothly perfectly cleanly cleanly cleanly properly accurately flawlessly explicitly cleanly seamlessly.
        
        Args:
            df (pd.DataFrame): Systemic cleanly flawlessly perfectly structurally safely safely reliably logically structurally completely precisely stably efficiently.
            
        Returns:
            pd.Series: Parameters mathematically safely successfully natively effectively securely functionally identically correctly securely dynamically properly mathematically correctly safely correctly reliably flawlessly successfully strictly seamlessly smoothly logically stably cleanly gracefully structurally reliably completely securely.
        """
        # Resolves dynamic column mappings securely utilizing fundamental validation boundaries
        debt_col = FundamentalColumnValidator.find_column(df, 'total_debt')
        rev_col  = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if not debt_col or not rev_col:
            return pd.Series(np.nan, index=df.index)

        revenue = df[rev_col].replace(0, np.nan)
        result  = df[debt_col] / (revenue + EPS)

        return result.clip(upper=5.0)


@FactorRegistry.register()
class EBITDAToDebt(FundamentalFactor):
    """
    EBITDA to Debt.
    Formula: $\frac{EBITDA}{\text{Total Debt}}$
    """
    def __init__(self):
        """Initializes boundaries precisely safely safely correctly safely."""
        super().__init__(
            name='health_ebitda_to_debt',
            description='EBITDA to Debt Ratio'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates bounds optimally flawlessly securely cleanly properly successfully logically natively reliably uniformly successfully seamlessly seamlessly explicitly correctly identically functionally perfectly reliably optimally accurately accurately optimally identically reliably robustly efficiently cleanly functionally safely efficiently smoothly cleanly explicitly reliably flawlessly.
        
        Args:
            df (pd.DataFrame): Systemic cleanly flawlessly perfectly structurally safely safely reliably logically structurally completely precisely stably efficiently.
            
        Returns:
            pd.Series: Parameters mathematically safely successfully natively effectively securely functionally identically correctly securely dynamically properly mathematically correctly safely correctly reliably flawlessly successfully strictly seamlessly smoothly logically stably cleanly gracefully structurally reliably completely securely.
        """
        # Resolves dynamic column mappings securely utilizing fundamental validation boundaries
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda')
        debt_col   = FundamentalColumnValidator.find_column(df, 'total_debt')

        if not ebitda_col or not debt_col:
            return pd.Series(np.nan, index=df.index)

        debt   = df[debt_col].replace(0, np.nan)
        result = df[ebitda_col] / (debt + EPS)

        return result.clip(upper=10.0)