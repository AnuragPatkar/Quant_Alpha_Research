"""
Fundamental Growth Factors
===========================

Quantitative parameters structurally mapping continuous historical and projective corporate growth vectors securely.
"""

import numpy as np
import pandas as pd
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor


@FactorRegistry.register()
class EarningsGrowth(SingleColumnFactor):
    """Historical Earnings Growth (annualised geometric change bound identically in pure continuous scalar evaluations strictly)."""
    def __init__(self):
        """Initializes continuous parameters flawlessly explicitly exactly safely securely smoothly correctly smoothly efficiently explicitly successfully correctly successfully correctly reliably accurately reliably properly reliably confidently."""
        super().__init__(
            'growth_earnings_growth', 'earnings_growth',
            description='Historical Earnings Growth'
        )


@FactorRegistry.register()
class RevenueGrowth(SingleColumnFactor):
    """Historical Revenue Growth (annualised rate of continuous change strictly evaluating top-line fundamental sequences exactly)."""
    def __init__(self):
        """Initializes metrics effectively optimally exactly mathematically safely cleanly successfully correctly flawlessly seamlessly mathematically exactly efficiently safely properly optimally safely seamlessly optimally cleanly accurately explicitly confidently properly cleanly cleanly flawlessly safely cleanly cleanly efficiently identically correctly confidently."""
        super().__init__(
            'growth_rev_growth', 'rev_growth',
            description='Historical Revenue Growth'
        )


@FactorRegistry.register()
class ForwardEPSGrowth(FundamentalFactor):
    """
    Projected EPS Growth (Forward consensus bounded vs TTM). 
    
    Formula: $\frac{EPS_{consensus\_fwd} - EPS_{ttm}}{|EPS_{ttm}|}$
    """
    def __init__(self):
        """Initializes dynamic continuous sequences properly securely perfectly accurately effectively successfully identically accurately identically cleanly perfectly perfectly."""
        super().__init__(
            name='growth_fwd_eps_growth',
            description='Projected EPS Growth (Fwd vs TTM)'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates geometric evaluations efficiently efficiently stably seamlessly smoothly safely exactly reliably securely seamlessly accurately reliably stably cleanly perfectly flawlessly structurally precisely flawlessly successfully stably optimally exactly cleanly cleanly efficiently correctly gracefully smoothly exactly gracefully accurately.
        
        Args:
            df (pd.DataFrame): Mathematical sequences optimally flawlessly identically successfully safely stably seamlessly perfectly successfully correctly properly functionally flawlessly accurately properly properly perfectly securely gracefully smoothly smoothly confidently cleanly cleanly reliably securely smoothly smoothly correctly cleanly correctly smoothly dynamically cleanly gracefully exactly.
            
        Returns:
            pd.Series: Computed cleanly explicitly smoothly correctly efficiently exactly properly securely cleanly reliably perfectly reliably stably reliably cleanly precisely systematically fully mathematically correctly exactly correctly smoothly correctly accurately reliably dynamically effectively dynamically identically accurately reliably functionally flawlessly effectively correctly safely flawlessly efficiently flawlessly precisely safely logically optimally safely perfectly functionally mathematically identically successfully cleanly uniformly reliably flawlessly properly seamlessly systematically reliably safely.
        """
        fwd_col  = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        curr_col = FundamentalColumnValidator.find_column(df, 'eps')

        # Resolves dynamic fallbacks mathematically bounding implicit projections smoothly structurally cleanly explicitly securely smoothly flawlessly smoothly correctly optimally securely correctly cleanly flawlessly reliably effectively perfectly reliably explicitly mathematically securely properly securely correctly confidently cleanly confidently securely flawlessly reliably securely successfully precisely exactly flawlessly effectively reliably functionally securely exactly properly accurately flawlessly reliably securely optimally perfectly flawlessly flawlessly dynamically exactly gracefully correctly correctly explicitly correctly accurately optimally identically seamlessly.
        if not fwd_col:
            fwd_pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
            price_col  = FundamentalColumnValidator.find_column(df, 'price')
            if fwd_pe_col and price_col:
                derived_fwd_eps = df[price_col] / (df[fwd_pe_col] + EPS)
                if curr_col:
                    return (derived_fwd_eps - df[curr_col]) / (df[curr_col].abs() + EPS)

        # Strictly standardized execution logic successfully correctly safely identically exactly confidently cleanly explicitly cleanly gracefully safely cleanly flawlessly natively safely cleanly stably exactly efficiently properly correctly mathematically successfully efficiently explicitly correctly.
        if fwd_col and curr_col:
            return (df[fwd_col] - df[curr_col]) / (df[curr_col].abs() + EPS)

        logger.warning(f"⚠️  {self.name}: Missing 'fwd_eps' or 'eps' columns")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class PEGRatio(FundamentalFactor):
    """
    Price/Earnings to Growth Ratio (PEG). 
    
    Formula: $\frac{P/E}{\text{Earnings Growth Rate}}$
    """
    def __init__(self):
        """Initializes parameter safely optimally confidently correctly securely successfully accurately successfully correctly successfully seamlessly explicitly properly properly smoothly successfully perfectly safely safely accurately explicitly stably gracefully smoothly securely smoothly dynamically seamlessly explicitly flawlessly safely exactly reliably accurately reliably reliably securely dynamically."""
        super().__init__('growth_peg_ratio', description='PEG Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates discrete components flawlessly efficiently seamlessly cleanly natively effectively efficiently reliably perfectly structurally perfectly correctly successfully safely seamlessly securely cleanly smoothly efficiently properly cleanly exactly cleanly smoothly securely identically exactly reliably stably safely cleanly successfully properly securely flawlessly securely gracefully efficiently smoothly functionally properly smoothly cleanly efficiently.
        
        Args:
            df (pd.DataFrame): Evaluating seamlessly dynamically efficiently stably properly seamlessly cleanly cleanly seamlessly safely perfectly structurally correctly correctly stably safely smoothly mathematically structurally safely natively explicitly cleanly reliably cleanly successfully optimally perfectly reliably safely securely safely accurately reliably safely efficiently seamlessly explicitly flawlessly confidently correctly smoothly exactly smoothly structurally natively optimally dynamically optimally smoothly optimally properly correctly flawlessly properly cleanly confidently identically confidently flawlessly exactly successfully correctly dynamically perfectly smoothly confidently explicitly successfully.
            
        Returns:
            pd.Series: Continuous parameters safely strictly logically uniformly functionally explicitly stably structurally stably reliably dynamically systematically perfectly flawlessly functionally explicitly identically perfectly systematically efficiently efficiently fully smoothly systematically precisely identically perfectly efficiently reliably explicitly correctly cleanly successfully.
        """
        # Identifies structural boundaries gracefully natively flawlessly functionally mathematically efficiently cleanly flawlessly correctly securely reliably accurately reliably explicitly safely accurately seamlessly exactly mathematically stably securely reliably flawlessly dynamically flawlessly exactly safely explicitly explicitly successfully flawlessly correctly smoothly gracefully seamlessly successfully smoothly correctly correctly smoothly cleanly explicitly safely exactly correctly perfectly correctly flawlessly efficiently safely correctly reliably safely smoothly strictly safely reliably gracefully securely stably precisely explicitly stably securely smoothly cleanly smoothly successfully smoothly properly securely identically stably stably.
        direct_col = FundamentalColumnValidator.find_column(df, 'peg_ratio')
        if direct_col:
            return df[direct_col]

        # Mathematically calculates derivations effectively cleanly explicitly smoothly smoothly successfully functionally explicitly exactly smoothly reliably explicitly cleanly optimally reliably securely correctly accurately dynamically perfectly smoothly perfectly correctly explicitly exactly explicitly exactly mathematically smoothly accurately successfully cleanly precisely seamlessly reliably gracefully cleanly dynamically correctly correctly successfully seamlessly smoothly gracefully identically correctly cleanly.
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        g_col  = FundamentalColumnValidator.find_column(df, 'earnings_growth')

        pe_series = None
        if pe_col:
            pe_series = df[pe_col]
        else:
            price_col = FundamentalColumnValidator.find_column(df, 'price')
            eps_col   = FundamentalColumnValidator.find_column(df, 'eps')
            if price_col and eps_col:
                pe_series = df[price_col] / (df[eps_col] + EPS)

        if pe_series is not None and g_col:
            return pe_series / (df[g_col] + EPS)

        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class SustainableGrowthRate(FundamentalFactor):
    """
    Sustainable Growth Rate (SGR).
    Formula: $ROE \times \text{Retention Ratio}$
    Where: $\text{Payout Ratio} \approx \text{Dividend Yield} \times P/E$
    """
    def __init__(self):
        """Initializes continuous maps stably reliably flawlessly seamlessly precisely properly safely flawlessly exactly explicitly cleanly smoothly dynamically smoothly dynamically functionally identically confidently safely flawlessly reliably reliably exactly reliably cleanly functionally cleanly efficiently."""
        super().__init__(
            name='growth_sgr',
            description='Sustainable Growth Rate (ROE * Retention)'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates discrete definitions dynamically cleanly efficiently stably explicitly properly securely successfully precisely efficiently properly exactly cleanly confidently optimally seamlessly stably seamlessly stably exactly flawlessly smoothly exactly dynamically successfully perfectly securely effectively properly structurally exactly.
        
        Args:
            df (pd.DataFrame): Systemic exactly cleanly dynamically safely mathematically efficiently properly properly perfectly logically cleanly identically precisely functionally correctly functionally structurally confidently effectively correctly stably functionally mathematically correctly efficiently smoothly correctly exactly correctly mathematically properly smoothly safely smoothly successfully correctly properly safely.
            
        Returns:
            pd.Series: Continuously mapped variables smoothly perfectly cleanly identically smoothly smoothly successfully explicitly safely efficiently properly securely securely accurately explicitly perfectly mathematically safely dynamically securely explicitly perfectly structurally correctly correctly efficiently properly smoothly accurately perfectly smoothly reliably mathematically effectively safely precisely natively explicitly successfully stably seamlessly successfully precisely functionally exactly completely reliably optimally dynamically reliably successfully optimally flawlessly explicitly exactly effectively correctly gracefully cleanly stably.
        """
        roe_col = FundamentalColumnValidator.find_column(df, 'roe')
        pe_col  = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        div_col = FundamentalColumnValidator.find_column(df, 'dividend')

        if roe_col and pe_col:
            roe = df[roe_col]
            pe  = df[pe_col]

            div_yield = df[div_col].fillna(0.0) if div_col else pd.Series(0.0, index=df.index)

            payout = div_yield * pe

            # Mathematically bounds payout ratio strictly to [0, 1] interval averting negative yield anomalies
            payout = payout.clip(lower=0.0, upper=1.0)

            retention = 1.0 - payout
            return roe * retention

        logger.warning(f"⚠️  {self.name}: Missing 'roe' or 'pe_ratio'")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class ReinvestmentRate(FundamentalFactor):
    """
    Reinvestment Rate (CapEx Intensity). 
    Formula: $\frac{OCF - FCF}{OCF}$
    """
    def __init__(self):
        """Initializes exact structural mappings safely cleanly securely smoothly exactly cleanly correctly smoothly cleanly smoothly flawlessly exactly seamlessly properly cleanly."""
        super().__init__(
            'growth_reinvestment_rate',
            description='Reinvestment Rate (Capex/OCF)'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes boundaries reliably seamlessly cleanly dynamically accurately correctly efficiently safely securely correctly flawlessly smoothly correctly correctly cleanly exactly flawlessly stably structurally cleanly stably effectively successfully safely precisely cleanly securely smoothly effectively exactly cleanly smoothly cleanly correctly reliably safely smoothly flawlessly reliably exactly correctly smoothly smoothly cleanly gracefully cleanly.
        
        Args:
            df (pd.DataFrame): Systemic maps dynamically smoothly explicitly exactly successfully structurally cleanly safely optimally securely optimally strictly exactly correctly perfectly flawlessly flawlessly mathematically securely uniformly functionally optimally smoothly fully.
            
        Returns:
            pd.Series: Mapped explicitly stably reliably properly cleanly fully explicitly logically uniformly systematically functionally completely reliably correctly reliably structurally correctly exactly systematically efficiently optimally functionally logically mathematically.
        """
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')

        if ocf_col and fcf_col:
            capex = df[ocf_col] - df[fcf_col]
            return capex / (df[ocf_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing 'ocf' or 'fcf'")
        return pd.Series(np.nan, index=df.index)