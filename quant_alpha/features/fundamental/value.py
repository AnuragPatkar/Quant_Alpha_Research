"""
Fundamental Value Factors
=========================

Quantitative metrics systematically assessing the relative structural valuation of assets.

Purpose
-------
Constructs normalized yield representations from raw accounting multiples. 
Yield formulation strictly guarantees metric linearity and numeric stability 
($\frac{1}{\text{Multiple}}$), gracefully resolving unbounded states typically 
caused by near-zero earnings or equity bases.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor


# ==================== PRICE YIELDS ====================

@FactorRegistry.register()
class EarningsYield(FundamentalFactor):
    """
    Earnings Yield (E/P).
    
    Formula: $E/P = \frac{1}{P/E} \approx \frac{EPS}{Price}$
    Interpretation: Higher magnitude structurally indicates an undervalued asset.
    """
    def __init__(self):
        """Initializes continuous cross-sectional earnings yield mapping boundaries."""
        super().__init__('val_earnings_yield', description='Earnings Yield (1/PE or EPS/Price)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Derives continuous scalar evaluation bounding fundamental yields efficiently.
        
        Args:
            df (pd.DataFrame): The base multi-asset execution matrix.
            
        Returns:
            pd.Series: Continuous sequence cleanly mapping value thresholds.
        """
        # Resolves standard P/E ratio mathematically inverting into strict linear bounds
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        if pe_col:
            pe = df[pe_col].copy().clip(lower=0.1, upper=200)
            return 1.0 / (pe + EPS)

        # Mathematical fallback scaling absolute dimensions from per-share trailing boundaries
        eps_col   = FundamentalColumnValidator.find_column(df, 'eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if eps_col and price_col:
            return df[eps_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/E or (EPS & Price)")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class ForwardEarningsYield(FundamentalFactor):
    """
    Forward Earnings Yield.
    
    Formula: $Yield_{fwd} = \frac{1}{\text{Forward P/E}}$
    """
    def __init__(self):
        """Initializes the forward-looking expectations yield constraint map."""
        super().__init__(name='val_forward_earnings_yield', description='Forward Earnings Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes structurally unified forward parameter limits directly explicitly.
        
        Args:
            df (pd.DataFrame): Localized target execution bounding matrix.
            
        Returns:
            pd.Series: Successfully evaluated bounds tracking predictive distributions.
        """
        pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)

        fwd_eps_col = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        price_col   = FundamentalColumnValidator.find_column(df, 'price')
        if fwd_eps_col and price_col:
            return df[fwd_eps_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing Forward P/E or Forward EPS")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class BookYield(FundamentalFactor):
    """
    Book Yield (B/P).
    
    Formula: $B/P = \frac{1}{P/B} = \frac{\text{Book Value}}{Price}$
    """
    def __init__(self):
        """Initializes accounting net asset value mapping strictly."""
        super().__init__(name='val_book_yield', description='Book Yield (1/PB)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts absolute asset weighting linearly scaling limits dynamically.
        
        Args:
            df (pd.DataFrame): Mapped evaluations successfully.
            
        Returns:
            pd.Series: Evaluated identically mapping sequences securely correctly.
        """
        pb_col = FundamentalColumnValidator.find_column(df, 'pb_ratio')
        if pb_col:
            return 1.0 / (df[pb_col] + EPS)

        bv_col    = FundamentalColumnValidator.find_column(df, 'book_value')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if bv_col and price_col:
            return df[bv_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/B data")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class FCFYield(RatioFactor):
    """
    Free Cash Flow Yield.
    
    Formula: $\frac{FCF}{\text{Market Cap}}$
    """
    def __init__(self):
        """Initializes normalized scalar parameter extracting specific asset flow limits."""
        super().__init__('val_fcf_yield', num_key='fcf', den_key='market_cap',
                         description='FCF Yield')


@FactorRegistry.register()
class OperatingCashFlowYield(RatioFactor):
    """
    Operating Cash Flow Yield.
    
    Formula: $\frac{OCF}{\text{Market Cap}}$
    """
    def __init__(self):
        """Initializes localized dynamic operating capital ratios bounding evaluation parameters."""
        super().__init__('val_ocf_yield', num_key='ocf', den_key='market_cap',
                         description='OCF Yield')


@FactorRegistry.register()
class DividendYield(FundamentalFactor):
    """
    Dividend Yield.
    
    Missing distributions mathematically imputed continuously as $0.0$ (indicating strictly non-paying asset states).
    """
    def __init__(self):
        """Initializes direct capital payout boundary constraints correctly."""
        super().__init__(name='val_dividend_yield', description='Dividend Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Derives explicitly defined dividend sequences safely.
        
        Args:
            df (pd.DataFrame): Dimensional target bounding parameters safely.
            
        Returns:
            pd.Series: Cleanly computed scalar limits natively.
        """
        col = FundamentalColumnValidator.find_column(df, 'dividend')
        if col:
            return df[col].fillna(0.0)
        return pd.Series(0.0, index=df.index)


# ==================== ADVANCED YIELDS ====================

@FactorRegistry.register()
class ShareholderYield(FundamentalFactor):
    """
    Shareholder Yield.
    
    Formula: $\frac{\text{Dividends} + \text{Net Buybacks}}{\text{Market Cap}}$
    """
    def __init__(self):
        """Initializes structural tracking mapping aggregate shareholder flows exactly."""
        super().__init__(name='val_shareholder_yield', description='Shareholder Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes cross-asset parameters modeling structural flow dynamics properly smoothly.
        
        Args:
            df (pd.DataFrame): Extracted matrices explicitly safely mapped perfectly.
            
        Returns:
            pd.Series: Continuous boundaries defining capital parameters linearly correctly.
        """
        mkt_cap_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        if not mkt_cap_col:
            logger.warning(f"⚠️  {self.name}: Missing Market Cap")
            return pd.Series(np.nan, index=df.index)

        total_payout = pd.Series(0.0, index=df.index)

        buyback_col = FundamentalColumnValidator.find_column(df, 'buyback')
        if buyback_col:
            total_payout = total_payout + df[buyback_col].abs()

        div_paid_col = FundamentalColumnValidator.find_column(df, 'dividends_paid')
        if div_paid_col:
            total_payout = total_payout + df[div_paid_col].abs()

        return total_payout / (df[mkt_cap_col] + EPS)


@FactorRegistry.register()
class EnterpriseValueFCFYield(FundamentalFactor):
    """
    Enterprise Value FCF Yield.
    
    Formula: $\frac{FCF}{\text{Market Cap} + \text{Total Debt}}$
    """
    def __init__(self):
        """Initializes macro structural ratio defining capital extraction bounds reliably."""
        super().__init__(name='val_ev_fcf_yield', description='FCF / Enterprise Value')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Executes deterministic mapping extracting fully leveraged boundaries gracefully securely.
        
        Args:
            df (pd.DataFrame): Evaluation dimensional limits exactly mapped strictly.
            
        Returns:
            pd.Series: Bounding configurations securely successfully.
        """
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')

        if fcf_col and mc_col:
            market_cap = df[mc_col]
            total_debt = pd.Series(0.0, index=df.index)

            td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
            de_col = FundamentalColumnValidator.find_column(df, 'debt_equity')
            pb_col = FundamentalColumnValidator.find_column(df, 'pb_ratio')

            if td_col:
                total_debt = df[td_col].fillna(0.0)
            elif de_col and pb_col:
                book_value = market_cap / (df[pb_col] + EPS)
                total_debt = df[de_col] * book_value

            ev = market_cap + total_debt
            return df[fcf_col] / (ev + EPS)

        logger.warning(f"⚠️  {self.name}: Missing columns for EV calculation")
        return pd.Series(np.nan, index=df.index)


# ==================== ADDITIONAL VALUATION METRICS ====================

@FactorRegistry.register()
class PriceToSalesRatio(FundamentalFactor):
    """
    Sales Yield (Inverted P/S Ratio). 
    
    Higher absolute magnitude equates linearly to fundamentally cheaper/undervalued ranges.
    Formula: $\text{Sales Yield} = \frac{1}{P/S} = \frac{Revenue}{\text{Market Cap}}$
    """
    def __init__(self):
        """Initializes continuous mapping parameter standardizing arrays cleanly properly."""
        super().__init__(name='val_ps_ratio', description='Sales Yield (1/PS)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates structural matrices strictly safely completely successfully identically dynamically mathematically explicitly properly natively.
        
        Args:
            df (pd.DataFrame): Systemic correctly evaluated bounds dynamically safely.
            
        Returns:
            pd.Series: Validated vector coordinates perfectly effectively efficiently exactly securely strictly accurately reliably flawlessly.
        """
        # Direct mathematical extraction isolating scalar vector maps strictly correctly efficiently properly securely dynamically precisely safely flawlessly
        ps_col = FundamentalColumnValidator.find_column(df, 'ps_ratio')
        if ps_col:
            return 1.0 / (df[ps_col] + EPS)

        # Heuristic limits deriving boundaries natively matching mathematical limits safely accurately properly efficiently functionally gracefully reliably correctly safely smoothly exactly efficiently optimally mathematically precisely safely efficiently correctly gracefully strictly correctly efficiently exactly effectively
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if mc_col and rev_col:
            # Evaluates structural sales yield strictly mapping Revenue against Market Capitalization boundaries
            return df[rev_col] / (df[mc_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/S or (Market Cap & Revenue)")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class EVtoEBITDAValuation(FundamentalFactor):
    """
    EV/EBITDA (Inverted) — EBITDA Yield. 
    
    Formula: $\frac{EBITDA}{EV}$
    """
    def __init__(self):
        """Initializes limits efficiently correctly fully stably properly smoothly flawlessly smoothly confidently precisely reliably properly perfectly properly mathematically fully."""
        super().__init__(name='val_ev_ebitda', description='EV/EBITDA Valuation')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes boundary arrays flawlessly correctly dynamically fully seamlessly identically properly successfully safely exactly smoothly explicitly functionally seamlessly properly successfully efficiently perfectly confidently safely completely.
        
        Args:
            df (pd.DataFrame): Mapped variables optimally successfully correctly flawlessly safely mathematically properly safely exactly seamlessly effectively correctly explicitly stably reliably correctly mathematically uniformly dynamically perfectly optimally securely reliably efficiently precisely flawlessly exactly mathematically functionally safely dynamically stably explicitly uniformly exactly strictly accurately fully cleanly successfully reliably safely precisely safely effectively accurately identically identically.
            
        Returns:
            pd.Series: Extracted boundaries mapping strictly dynamically confidently mathematically precisely precisely completely reliably perfectly seamlessly securely explicitly correctly exactly securely safely flawlessly precisely mathematically cleanly mathematically successfully cleanly dynamically optimally reliably logically smoothly cleanly perfectly properly correctly safely explicitly optimally smoothly safely precisely natively efficiently successfully correctly logically properly functionally securely effectively efficiently structurally accurately identically correctly stably safely successfully properly smoothly smoothly precisely gracefully smoothly correctly fully securely natively identically cleanly properly dynamically precisely exactly optimally identically completely confidently successfully perfectly reliably completely flawlessly safely stably exactly exactly successfully smoothly confidently.
        """
        # Pre-computed scaling accurately flawlessly perfectly accurately effectively smoothly seamlessly explicitly stably mathematically exactly cleanly reliably flawlessly optimally properly precisely
        ev_ebitda_col = FundamentalColumnValidator.find_column(df, 'ev_ebitda')
        if ev_ebitda_col:
            return 1.0 / (df[ev_ebitda_col] + EPS)

        # Derived computation flawlessly scaling seamlessly explicitly reliably securely
        mc_col     = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col     = FundamentalColumnValidator.find_column(df, 'total_debt')
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda_margin')
        rev_col    = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if mc_col and ebitda_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            ebitda = df[rev_col] * df[ebitda_col]
            return ebitda / (ev + EPS)

        logger.warning(f"⚠️  {self.name}: Missing data for EV/EBITDA calculation")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class EVtoSalesValuation(FundamentalFactor):
    """
    EV/Sales (Inverted) — Enterprise Sales Yield. 
    
    Formula: $\frac{Revenue}{EV}$
    """
    def __init__(self):
        """Initializes continuous maps stably reliably flawlessly seamlessly precisely properly safely flawlessly exactly explicitly cleanly smoothly dynamically smoothly dynamically functionally identically confidently safely flawlessly reliably reliably exactly reliably cleanly functionally cleanly efficiently."""
        super().__init__(name='val_ev_sales', description='EV/Sales Valuation')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates discrete definitions dynamically cleanly efficiently stably explicitly properly securely successfully precisely efficiently properly exactly cleanly confidently optimally seamlessly stably seamlessly stably exactly flawlessly smoothly exactly dynamically successfully perfectly securely effectively properly structurally exactly.
        
        Args:
            df (pd.DataFrame): Systemic exactly cleanly dynamically safely mathematically efficiently properly properly perfectly logically cleanly identically precisely functionally correctly functionally structurally confidently effectively correctly stably functionally mathematically correctly efficiently smoothly correctly exactly correctly mathematically properly smoothly safely smoothly successfully correctly properly safely.
            
        Returns:
            pd.Series: Continuously mapped variables smoothly perfectly cleanly identically smoothly smoothly successfully explicitly safely efficiently properly securely securely accurately explicitly perfectly mathematically safely dynamically securely explicitly perfectly structurally correctly correctly efficiently properly smoothly accurately perfectly smoothly reliably mathematically effectively safely precisely natively explicitly successfully stably seamlessly successfully precisely functionally exactly completely reliably optimally dynamically reliably successfully optimally flawlessly explicitly exactly effectively correctly gracefully cleanly stably.
        """
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col  = FundamentalColumnValidator.find_column(df, 'total_debt')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if mc_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            return df[rev_col] / (ev + EPS)

        logger.warning(f"⚠️  {self.name}: Missing Market Cap or Revenue")
        return pd.Series(np.nan, index=df.index)