"""
NARDL Variable Decomposition Tool
Based on Shin, Yu & Greenwood-Nimmo (2014)
"Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework"

Features:
- Variable decomposition into positive/negative partial sums
- Unit Root Tests: ADF, PP (Phillips-Perron), KPSS
- Interactive Plotly visualizations
- Descriptive statistics
- Excel export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import arch for Phillips-Perron test
try:
    from arch.unitroot import PhillipsPerron
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="NARDL Decomposition Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette - Modern professional colors
COLORS = {
    'primary': '#2E86AB',      # Steel Blue
    'secondary': '#A23B72',    # Raspberry
    'positive': '#28A745',     # Green for positive
    'negative': '#DC3545',     # Red for negative
    'original': '#6C757D',     # Gray for original
    'accent1': '#F18F01',      # Orange
    'accent2': '#C73E1D',      # Dark Red
    'accent3': '#3A0CA3',      # Deep Purple
    'background': '#F8F9FA',   # Light gray
    'text': '#212529',         # Dark text
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .info-box {
        background-color: #E7F3FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D4EDDA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28A745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F18F01;
        margin: 1rem 0;
    }
    .test-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E0E0E0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


def decompose_variable(series: pd.Series, threshold: float = 0.0) -> tuple:
    """
    Decompose a variable into positive and negative partial sums
    Based on Shin, Yu & Greenwood-Nimmo (2014) methodology
    
    x_t = x_0 + x_t^+ + x_t^-
    
    where:
    x_t^+ = Σ max(Δx_j, 0) - partial sum of positive changes
    x_t^- = Σ min(Δx_j, 0) - partial sum of negative changes
    """
    # Calculate first differences (changes)
    delta_x = series.diff()
    
    # Calculate positive changes (above threshold)
    delta_x_pos = delta_x.apply(lambda x: max(x - threshold, 0) if pd.notna(x) else np.nan)
    
    # Calculate negative changes (below threshold)
    delta_x_neg = delta_x.apply(lambda x: min(x - threshold, 0) if pd.notna(x) else np.nan)
    
    # Calculate cumulative sums (partial sums)
    x_pos = delta_x_pos.cumsum()
    x_neg = delta_x_neg.cumsum()
    
    # Set first value to 0 (or initial value consideration)
    x_pos.iloc[0] = 0
    x_neg.iloc[0] = 0
    
    return x_pos, x_neg, delta_x, delta_x_pos, delta_x_neg


def perform_adf_test(series: pd.Series, regression: str = 'c', maxlag: int = None, autolag: str = 'AIC') -> dict:
    """
    Perform Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pd.Series - Time series data
    regression : str - 'c' (constant), 'ct' (constant+trend), 'ctt' (constant+linear+quadratic trend), 'n' (none)
    maxlag : int - Maximum lag to consider
    autolag : str - Method for automatic lag selection ('AIC', 'BIC', 't-stat', None)
    """
    clean_series = series.dropna()
    if len(clean_series) < 20:
        return {'error': 'Insufficient observations (need at least 20)'}
    
    try:
        if autolag == 'None':
            autolag = None
        result = adfuller(clean_series, regression=regression, maxlag=maxlag, autolag=autolag)
        return {
            'Test Statistic': result[0],
            'P-Value': result[1],
            'Lags Used': result[2],
            'Observations': result[3],
            'Critical Values': result[4],
            'Is Stationary (5%)': result[0] < result[4]['5%']
        }
    except Exception as e:
        return {'error': str(e)}


def perform_pp_test(series: pd.Series, trend: str = 'c', lags: int = None, test_type: str = 'tau') -> dict:
    """
    Perform Phillips-Perron test
    
    Parameters:
    -----------
    series : pd.Series - Time series data
    trend : str - 'n' (no constant), 'c' (constant only), 'ct' (constant and trend)
    lags : int - Number of lags for Newey-West covariance estimation (None for automatic)
    test_type : str - 'tau' (t-stat based) or 'rho' (regression coefficient based)
    """
    if not ARCH_AVAILABLE:
        return {'error': 'arch library not installed. Install with: pip install arch'}
    
    clean_series = series.dropna()
    if len(clean_series) < 20:
        return {'error': 'Insufficient observations (need at least 20)'}
    
    try:
        pp = PhillipsPerron(clean_series, trend=trend, lags=lags, test_type=test_type)
        return {
            'Test Statistic': pp.stat,
            'P-Value': pp.pvalue,
            'Lags Used': pp.lags,
            'Observations': pp.nobs,
            'Critical Values': {
                '1%': pp.critical_values['1%'],
                '5%': pp.critical_values['5%'],
                '10%': pp.critical_values['10%']
            },
            'Is Stationary (5%)': pp.stat < pp.critical_values['5%']
        }
    except Exception as e:
        return {'error': str(e)}


def perform_kpss_test(series: pd.Series, regression: str = 'c', nlags: str = 'auto') -> dict:
    """
    Perform KPSS test
    
    Parameters:
    -----------
    series : pd.Series - Time series data
    regression : str - 'c' (level stationarity) or 'ct' (trend stationarity)
    nlags : str/int - Number of lags ('auto', 'legacy', or integer)
    """
    clean_series = series.dropna()
    if len(clean_series) < 20:
        return {'error': 'Insufficient observations (need at least 20)'}
    
    try:
        result = kpss(clean_series, regression=regression, nlags=nlags)
        return {
            'Test Statistic': result[0],
            'P-Value': result[1],
            'Lags Used': result[2],
            'Critical Values': result[3],
            'Is Stationary (5%)': result[0] < result[3]['5%']
        }
    except Exception as e:
        return {'error': str(e)}


def calculate_descriptive_stats(series: pd.Series) -> dict:
    """Calculate comprehensive descriptive statistics"""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return {}
    
    stats_dict = {
        'Count': len(clean_series),
        'Mean': clean_series.mean(),
        'Median': clean_series.median(),
        'Std Dev': clean_series.std(),
        'Variance': clean_series.var(),
        'Min': clean_series.min(),
        'Max': clean_series.max(),
        'Range': clean_series.max() - clean_series.min(),
        'Skewness': clean_series.skew(),
        'Kurtosis': clean_series.kurtosis(),
        'Q1 (25%)': clean_series.quantile(0.25),
        'Q3 (75%)': clean_series.quantile(0.75),
        'IQR': clean_series.quantile(0.75) - clean_series.quantile(0.25),
        'CV (%)': (clean_series.std() / clean_series.mean() * 100) if clean_series.mean() != 0 else np.nan
    }
    return stats_dict


def create_decomposition_plot(original: pd.Series, pos: pd.Series, neg: pd.Series, 
                             var_name: str, index=None) -> go.Figure:
    """Create interactive decomposition visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Original Series: {var_name}',
            f'Positive & Negative Partial Sums',
            f'First Differences (Δ{var_name})',
            f'Decomposition Components'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    x_axis = index if index is not None else list(range(len(original)))
    
    # Original series
    fig.add_trace(
        go.Scatter(x=x_axis, y=original, mode='lines', 
                  name='Original', line=dict(color=COLORS['original'], width=2)),
        row=1, col=1
    )
    
    # Positive and negative partial sums
    fig.add_trace(
        go.Scatter(x=x_axis, y=pos, mode='lines', 
                  name=f'{var_name}⁺ (Positive)', line=dict(color=COLORS['positive'], width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=neg, mode='lines', 
                  name=f'{var_name}⁻ (Negative)', line=dict(color=COLORS['negative'], width=2)),
        row=1, col=2
    )
    
    # First differences
    delta_x = original.diff()
    colors = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in delta_x.fillna(0)]
    fig.add_trace(
        go.Bar(x=x_axis, y=delta_x, name='Changes', marker_color=colors),
        row=2, col=1
    )
    
    # Stacked area for decomposition
    fig.add_trace(
        go.Scatter(x=x_axis, y=pos, mode='lines', fill='tozeroy',
                  name='Cumulative Positive', fillcolor='rgba(40, 167, 69, 0.3)',
                  line=dict(color=COLORS['positive'], width=1)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=neg, mode='lines', fill='tozeroy',
                  name='Cumulative Negative', fillcolor='rgba(220, 53, 69, 0.3)',
                  line=dict(color=COLORS['negative'], width=1)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        template='plotly_white',
        font=dict(family="Arial", size=11),
        title=dict(text=f'NARDL Decomposition Analysis: {var_name}', 
                  font=dict(size=16, color=COLORS['text']))
    )
    
    return fig


def create_comparison_plot(decomposed_data: dict, variable_names: list) -> go.Figure:
    """Create comparison plot for multiple decomposed variables"""
    n_vars = len(variable_names)
    
    fig = make_subplots(
        rows=n_vars, cols=2,
        subplot_titles=[f'{v}⁺' if i % 2 == 0 else f'{v}⁻' 
                       for v in variable_names for i in range(2)],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    for idx, var in enumerate(variable_names):
        row = idx + 1
        pos_col = f'{var}_pos'
        neg_col = f'{var}_neg'
        
        if pos_col in decomposed_data.columns:
            fig.add_trace(
                go.Scatter(y=decomposed_data[pos_col], mode='lines',
                          name=f'{var}⁺', line=dict(color=COLORS['positive'], width=2)),
                row=row, col=1
            )
        if neg_col in decomposed_data.columns:
            fig.add_trace(
                go.Scatter(y=decomposed_data[neg_col], mode='lines',
                          name=f'{var}⁻', line=dict(color=COLORS['negative'], width=2)),
                row=row, col=2
            )
    
    fig.update_layout(
        height=250 * n_vars,
        showlegend=False,
        template='plotly_white',
        title=dict(text='Positive & Negative Partial Sums Comparison',
                  font=dict(size=16, color=COLORS['text']))
    )
    
    return fig


def to_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Decomposed_Data')
    return output.getvalue()


def to_excel_multiple_sheets(dfs: dict) -> bytes:
    """Convert multiple DataFrames to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, index=True, sheet_name=sheet_name[:31])  # Excel sheet name limit
    return output.getvalue()


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 NARDL Variable Decomposition Tool</h1>', unsafe_allow_html=True)
    st.markdown('''<p class="sub-header">
        Based on Shin, Yu & Greenwood-Nimmo (2014)<br>
        <em>"Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework"</em>
    </p>''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your Excel file (.xlsx)",
            type=['xlsx', 'xls'],
            help="Upload a time series dataset in Excel format"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About NARDL Decomposition")
        st.markdown("""
        The NARDL methodology decomposes a variable **x** into:
        
        - **x⁺**: Cumulative sum of positive changes
        - **x⁻**: Cumulative sum of negative changes
        
        This allows modeling **asymmetric effects** in both short-run and long-run relationships.
        """)
        
        with st.expander("📚 Mathematical Formula"):
            st.latex(r"x_t = x_0 + x_t^+ + x_t^-")
            st.latex(r"x_t^+ = \sum_{j=1}^{t} \max(\Delta x_j, 0)")
            st.latex(r"x_t^- = \sum_{j=1}^{t} \min(\Delta x_j, 0)")
        
        st.markdown("---")
        st.markdown("### 🧪 Unit Root Tests Available")
        st.markdown("""
        - **ADF** - Augmented Dickey-Fuller
        - **PP** - Phillips-Perron
        - **KPSS** - Kwiatkowski-Phillips-Schmidt-Shin
        """)
        
        if not ARCH_AVAILABLE:
            st.warning("⚠️ `arch` library not installed. Phillips-Perron test unavailable.")
    
    # Main content
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_excel(uploaded_file)
            
            # Display data info
            st.markdown("### 📋 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Observations", df.shape[0])
            with col2:
                st.metric("Variables", df.shape[1])
            with col3:
                st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
            with col4:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", missing)
            
            # Show data preview
            with st.expander("👁️ Preview Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Variable selection
            st.markdown("---")
            st.markdown("### ⚙️ Variable Selection")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔄 Variables to Decompose")
                decompose_vars = st.multiselect(
                    "Select variables to decompose into positive and negative partial sums",
                    options=numeric_cols,
                    help="These variables will be decomposed using the NARDL methodology"
                )
            
            with col2:
                st.markdown("#### 📌 Non-Decomposed Variables")
                remaining_vars = [v for v in all_cols if v not in decompose_vars]
                non_decompose_vars = st.multiselect(
                    "Select variables to keep without decomposition",
                    options=remaining_vars,
                    help="These variables will be included in the output without transformation"
                )
            
            # Decomposition threshold
            st.markdown("#### 🎯 Decomposition Settings")
            threshold = st.number_input(
                "Threshold for decomposition (default: 0)",
                value=0.0,
                step=0.01,
                help="Changes above this threshold are positive, below are negative"
            )
            
            # Time/Index column selection
            index_col = st.selectbox(
                "Select time/index column (optional)",
                options=['None'] + all_cols,
                help="Select a column to use as the time index for visualization"
            )
            
            if decompose_vars:
                st.markdown("---")
                st.markdown("### 🔬 Decomposition Results")
                
                # Perform decomposition
                decomposed_data = pd.DataFrame()
                
                # Set index if selected
                if index_col != 'None':
                    decomposed_data.index = df[index_col]
                
                # Add non-decomposed variables
                for var in non_decompose_vars:
                    decomposed_data[var] = df[var].values
                
                # Decompose selected variables
                decomposition_results = {}
                for var in decompose_vars:
                    series = df[var]
                    x_pos, x_neg, delta_x, delta_pos, delta_neg = decompose_variable(series, threshold)
                    
                    decomposed_data[var] = series.values
                    decomposed_data[f'{var}_pos'] = x_pos.values
                    decomposed_data[f'{var}_neg'] = x_neg.values
                    decomposed_data[f'{var}_delta'] = delta_x.values
                    decomposed_data[f'{var}_delta_pos'] = delta_pos.values
                    decomposed_data[f'{var}_delta_neg'] = delta_neg.values
                    
                    decomposition_results[var] = {
                        'original': series,
                        'positive': x_pos,
                        'negative': x_neg,
                        'delta': delta_x,
                        'delta_pos': delta_pos,
                        'delta_neg': delta_neg
                    }
                
                # Tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Visualizations", 
                    "📊 Descriptive Statistics", 
                    "🧪 Unit Root Tests",
                    "📥 Download Data"
                ])
                
                # TAB 1: Visualizations
                with tab1:
                    st.markdown("#### Decomposition Plots")
                    
                    for var in decompose_vars:
                        with st.expander(f"📊 {var} Decomposition", expanded=True):
                            results = decomposition_results[var]
                            fig = create_decomposition_plot(
                                results['original'],
                                results['positive'],
                                results['negative'],
                                var,
                                decomposed_data.index if index_col != 'None' else None
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparison plot
                    if len(decompose_vars) > 1:
                        st.markdown("#### Multi-Variable Comparison")
                        fig_comp = create_comparison_plot(decomposed_data, decompose_vars)
                        st.plotly_chart(fig_comp, use_container_width=True)
                
                # TAB 2: Descriptive Statistics
                with tab2:
                    st.markdown("#### Comprehensive Descriptive Statistics")
                    
                    # Statistics for original variables
                    st.markdown("##### Original Variables")
                    stats_original = {}
                    for var in decompose_vars:
                        stats_original[var] = calculate_descriptive_stats(df[var])
                    
                    stats_df_orig = pd.DataFrame(stats_original).T
                    st.dataframe(stats_df_orig.style.format("{:.4f}"), use_container_width=True)
                    
                    # Statistics for decomposed variables
                    st.markdown("##### Positive Partial Sums (x⁺)")
                    stats_pos = {}
                    for var in decompose_vars:
                        stats_pos[f'{var}⁺'] = calculate_descriptive_stats(decomposed_data[f'{var}_pos'])
                    
                    stats_df_pos = pd.DataFrame(stats_pos).T
                    st.dataframe(stats_df_pos.style.format("{:.4f}"), use_container_width=True)
                    
                    st.markdown("##### Negative Partial Sums (x⁻)")
                    stats_neg = {}
                    for var in decompose_vars:
                        stats_neg[f'{var}⁻'] = calculate_descriptive_stats(decomposed_data[f'{var}_neg'])
                    
                    stats_df_neg = pd.DataFrame(stats_neg).T
                    st.dataframe(stats_df_neg.style.format("{:.4f}"), use_container_width=True)
                    
                    # First differences statistics
                    st.markdown("##### First Differences (Δx)")
                    stats_delta = {}
                    for var in decompose_vars:
                        stats_delta[f'Δ{var}'] = calculate_descriptive_stats(decomposed_data[f'{var}_delta'])
                    
                    stats_df_delta = pd.DataFrame(stats_delta).T
                    st.dataframe(stats_df_delta.style.format("{:.4f}"), use_container_width=True)
                
                # TAB 3: Unit Root Tests
                with tab3:
                    st.markdown("#### 🔧 Unit Root Test Configuration")
                    
                    # Test selection
                    st.markdown("##### Select Tests to Run")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        run_adf = st.checkbox("ADF Test", value=True, help="Augmented Dickey-Fuller test")
                    with col2:
                        run_pp = st.checkbox("Phillips-Perron Test", value=ARCH_AVAILABLE, 
                                           disabled=not ARCH_AVAILABLE,
                                           help="Phillips-Perron test (requires arch library)")
                    with col3:
                        run_kpss = st.checkbox("KPSS Test", value=True, help="KPSS stationarity test")
                    
                    st.markdown("---")
                    
                    # ===== ADF TEST OPTIONS =====
                    if run_adf:
                        st.markdown("##### 📊 ADF Test Options")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            adf_regression = st.selectbox(
                                "Regression Type",
                                options=['c', 'ct', 'ctt', 'n'],
                                format_func=lambda x: {
                                    'c': 'Constant only',
                                    'ct': 'Constant + Trend',
                                    'ctt': 'Constant + Linear + Quadratic Trend',
                                    'n': 'No constant, No trend'
                                }[x],
                                help="Specify the deterministic terms in the ADF regression",
                                key="adf_reg"
                            )
                        
                        with col2:
                            adf_maxlag = st.number_input(
                                "Maximum Lag",
                                min_value=1, max_value=50, value=12,
                                help="Maximum lag order to consider. Rule of thumb: 12*(T/100)^(1/4)",
                                key="adf_maxlag"
                            )
                        
                        with col3:
                            adf_autolag = st.selectbox(
                                "Lag Selection Method",
                                options=['AIC', 'BIC', 't-stat', 'None'],
                                help="Method for automatic lag length selection. 'None' uses maxlag.",
                                key="adf_autolag"
                            )
                    
                    # ===== PHILLIPS-PERRON TEST OPTIONS =====
                    if run_pp and ARCH_AVAILABLE:
                        st.markdown("##### 📊 Phillips-Perron Test Options")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pp_trend = st.selectbox(
                                "Trend Type",
                                options=['c', 'ct', 'n'],
                                format_func=lambda x: {
                                    'c': 'Constant only',
                                    'ct': 'Constant + Trend',
                                    'n': 'No constant, No trend'
                                }[x],
                                help="Deterministic trend terms",
                                key="pp_trend"
                            )
                        
                        with col2:
                            pp_lags_auto = st.checkbox("Auto-select lags (Newey-West)", value=True, key="pp_auto")
                            if not pp_lags_auto:
                                pp_lags = st.number_input(
                                    "Number of Lags (Newey-West)",
                                    min_value=0, max_value=50, value=4,
                                    help="Lags for Newey-West HAC covariance estimation",
                                    key="pp_lags"
                                )
                            else:
                                pp_lags = None
                        
                        with col3:
                            pp_test_type = st.selectbox(
                                "Test Statistic Type",
                                options=['tau', 'rho'],
                                format_func=lambda x: {
                                    'tau': 'Z-tau (t-statistic based)',
                                    'rho': 'Z-rho (coefficient based)'
                                }[x],
                                help="Type of test statistic to compute",
                                key="pp_type"
                            )
                    
                    # ===== KPSS TEST OPTIONS =====
                    if run_kpss:
                        st.markdown("##### 📊 KPSS Test Options")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            kpss_regression = st.selectbox(
                                "Null Hypothesis Type",
                                options=['c', 'ct'],
                                format_func=lambda x: {
                                    'c': 'Level Stationarity (constant only)',
                                    'ct': 'Trend Stationarity (constant + trend)'
                                }[x],
                                help="Type of stationarity to test",
                                key="kpss_reg"
                            )
                        
                        with col2:
                            kpss_lags = st.selectbox(
                                "Lag Selection",
                                options=['auto', 'legacy'],
                                format_func=lambda x: {
                                    'auto': 'Automatic (Schwert formula)',
                                    'legacy': 'Legacy (int(12*(T/100)^(1/4)))'
                                }[x],
                                help="Method for selecting bandwidth/lags",
                                key="kpss_lags"
                            )
                    
                    st.markdown("---")
                    
                    # RUN TESTS BUTTON
                    if st.button("🔬 Run Unit Root Tests", type="primary", use_container_width=True):
                        st.markdown("---")
                        
                        # Prepare test variables
                        test_variables = []
                        for var in decompose_vars:
                            test_variables.extend([
                                (var, decomposed_data[var] if var in decomposed_data.columns else df[var]),
                                (f'{var}⁺', decomposed_data[f'{var}_pos']),
                                (f'{var}⁻', decomposed_data[f'{var}_neg']),
                                (f'Δ{var}', decomposed_data[f'{var}_delta'])
                            ])
                        
                        # ===== ADF TEST RESULTS =====
                        if run_adf:
                            st.markdown("### 📊 Augmented Dickey-Fuller (ADF) Test Results")
                            st.markdown("""
                            <div class="info-box">
                            <b>H₀:</b> The series has a unit root (non-stationary)<br>
                            <b>H₁:</b> The series is stationary<br>
                            <b>Decision:</b> Reject H₀ if p-value < 0.05 (or test statistic < critical value)
                            </div>
                            """, unsafe_allow_html=True)
                            
                            adf_results = []
                            for name, series in test_variables:
                                result = perform_adf_test(
                                    series, 
                                    regression=adf_regression, 
                                    maxlag=adf_maxlag,
                                    autolag=adf_autolag
                                )
                                if 'error' not in result:
                                    adf_results.append({
                                        'Variable': name,
                                        'Test Statistic': result['Test Statistic'],
                                        'P-Value': result['P-Value'],
                                        'Lags Used': result['Lags Used'],
                                        'Critical 1%': result['Critical Values']['1%'],
                                        'Critical 5%': result['Critical Values']['5%'],
                                        'Critical 10%': result['Critical Values']['10%'],
                                        'Stationary (5%)': '✅ Yes' if result['Is Stationary (5%)'] else '❌ No'
                                    })
                                else:
                                    st.warning(f"ADF test error for {name}: {result['error']}")
                            
                            if adf_results:
                                adf_df = pd.DataFrame(adf_results)
                                st.dataframe(
                                    adf_df.style.format({
                                        'Test Statistic': '{:.4f}',
                                        'P-Value': '{:.4f}',
                                        'Critical 1%': '{:.4f}',
                                        'Critical 5%': '{:.4f}',
                                        'Critical 10%': '{:.4f}'
                                    }),
                                    use_container_width=True
                                )
                        
                        # ===== PHILLIPS-PERRON TEST RESULTS =====
                        if run_pp and ARCH_AVAILABLE:
                            st.markdown("### 📊 Phillips-Perron (PP) Test Results")
                            st.markdown("""
                            <div class="info-box">
                            <b>H₀:</b> The series has a unit root (non-stationary)<br>
                            <b>H₁:</b> The series is stationary<br>
                            <b>Advantage:</b> Robust to serial correlation and heteroskedasticity without adding lagged differences<br>
                            <b>Decision:</b> Reject H₀ if p-value < 0.05
                            </div>
                            """, unsafe_allow_html=True)
                            
                            pp_results = []
                            for name, series in test_variables:
                                result = perform_pp_test(
                                    series,
                                    trend=pp_trend,
                                    lags=pp_lags if not pp_lags_auto else None,
                                    test_type=pp_test_type
                                )
                                if 'error' not in result:
                                    pp_results.append({
                                        'Variable': name,
                                        'Test Statistic': result['Test Statistic'],
                                        'P-Value': result['P-Value'],
                                        'Lags (NW)': result['Lags Used'],
                                        'Critical 1%': result['Critical Values']['1%'],
                                        'Critical 5%': result['Critical Values']['5%'],
                                        'Critical 10%': result['Critical Values']['10%'],
                                        'Stationary (5%)': '✅ Yes' if result['Is Stationary (5%)'] else '❌ No'
                                    })
                                else:
                                    st.warning(f"PP test error for {name}: {result['error']}")
                            
                            if pp_results:
                                pp_df = pd.DataFrame(pp_results)
                                st.dataframe(
                                    pp_df.style.format({
                                        'Test Statistic': '{:.4f}',
                                        'P-Value': '{:.4f}',
                                        'Critical 1%': '{:.4f}',
                                        'Critical 5%': '{:.4f}',
                                        'Critical 10%': '{:.4f}'
                                    }),
                                    use_container_width=True
                                )
                        
                        # ===== KPSS TEST RESULTS =====
                        if run_kpss:
                            st.markdown("### 📊 KPSS Test Results")
                            st.markdown("""
                            <div class="info-box">
                            <b>H₀:</b> The series is stationary<br>
                            <b>H₁:</b> The series has a unit root (non-stationary)<br>
                            <b>Note:</b> KPSS tests the OPPOSITE null hypothesis compared to ADF and PP<br>
                            <b>Decision:</b> Reject H₀ if test statistic > critical value
                            </div>
                            """, unsafe_allow_html=True)
                            
                            kpss_results = []
                            for name, series in test_variables:
                                result = perform_kpss_test(
                                    series, 
                                    regression=kpss_regression,
                                    nlags=kpss_lags
                                )
                                if 'error' not in result:
                                    kpss_results.append({
                                        'Variable': name,
                                        'Test Statistic': result['Test Statistic'],
                                        'P-Value': result['P-Value'],
                                        'Lags Used': result['Lags Used'],
                                        'Critical 1%': result['Critical Values']['1%'],
                                        'Critical 5%': result['Critical Values']['5%'],
                                        'Critical 10%': result['Critical Values']['10%'],
                                        'Stationary (5%)': '✅ Yes' if result['Is Stationary (5%)'] else '❌ No'
                                    })
                                else:
                                    st.warning(f"KPSS test error for {name}: {result['error']}")
                            
                            if kpss_results:
                                kpss_df = pd.DataFrame(kpss_results)
                                st.dataframe(
                                    kpss_df.style.format({
                                        'Test Statistic': '{:.4f}',
                                        'P-Value': '{:.4f}',
                                        'Critical 1%': '{:.4f}',
                                        'Critical 5%': '{:.4f}',
                                        'Critical 10%': '{:.4f}'
                                    }),
                                    use_container_width=True
                                )
                        
                        # ===== SUMMARY INTERPRETATION =====
                        st.markdown("### 📝 Test Interpretation Summary")
                        st.markdown("""
                        <div class="info-box">
                        <b>Best Practice:</b> Compare results from multiple tests. If ADF/PP reject H₀ (stationary) 
                        and KPSS fails to reject H₀ (stationary), there is strong evidence for stationarity.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for name, series in test_variables:
                            conclusions = []
                            
                            if run_adf:
                                adf_result = perform_adf_test(series, regression=adf_regression, 
                                                            maxlag=adf_maxlag, autolag=adf_autolag)
                                if 'error' not in adf_result:
                                    conclusions.append(('ADF', adf_result['Is Stationary (5%)']))
                            
                            if run_pp and ARCH_AVAILABLE:
                                pp_result = perform_pp_test(series, trend=pp_trend,
                                                          lags=pp_lags if not pp_lags_auto else None,
                                                          test_type=pp_test_type)
                                if 'error' not in pp_result:
                                    conclusions.append(('PP', pp_result['Is Stationary (5%)']))
                            
                            if run_kpss:
                                kpss_result = perform_kpss_test(series, regression=kpss_regression,
                                                               nlags=kpss_lags)
                                if 'error' not in kpss_result:
                                    conclusions.append(('KPSS', kpss_result['Is Stationary (5%)']))
                            
                            if conclusions:
                                stationary_count = sum(1 for _, is_stat in conclusions if is_stat)
                                total_tests = len(conclusions)
                                
                                if stationary_count == total_tests:
                                    conclusion = "🟢 **Stationary** (all tests agree)"
                                    color = "success-box"
                                elif stationary_count == 0:
                                    conclusion = "🔴 **Non-stationary** (all tests agree)"
                                    color = "warning-box"
                                else:
                                    test_details = ", ".join([f"{t}: {'✓' if s else '✗'}" for t, s in conclusions])
                                    conclusion = f"🟡 **Mixed results** ({test_details})"
                                    color = "info-box"
                                
                                st.markdown(f"""
                                <div class="{color}">
                                <b>{name}:</b> {conclusion}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Download test results
                        st.markdown("---")
                        st.markdown("##### 📥 Download Test Results")
                        
                        test_results_sheets = {}
                        if run_adf and adf_results:
                            test_results_sheets['ADF_Results'] = pd.DataFrame(adf_results)
                        if run_pp and ARCH_AVAILABLE and pp_results:
                            test_results_sheets['PP_Results'] = pd.DataFrame(pp_results)
                        if run_kpss and kpss_results:
                            test_results_sheets['KPSS_Results'] = pd.DataFrame(kpss_results)
                        
                        if test_results_sheets:
                            test_excel = to_excel_multiple_sheets(test_results_sheets)
                            st.download_button(
                                label="📥 Download All Test Results (Excel)",
                                data=test_excel,
                                file_name="unit_root_test_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                # TAB 4: Download Data
                with tab4:
                    st.markdown("#### Download Decomposed Data")
                    
                    # Preview decomposed data
                    st.markdown("##### Data Preview")
                    st.dataframe(decomposed_data.head(20), use_container_width=True)
                    
                    # Column information
                    st.markdown("##### Column Description")
                    col_desc = []
                    for var in non_decompose_vars:
                        col_desc.append({'Column': var, 'Description': 'Original variable (non-decomposed)'})
                    for var in decompose_vars:
                        col_desc.extend([
                            {'Column': var, 'Description': f'Original {var} series'},
                            {'Column': f'{var}_pos', 'Description': f'Positive partial sum of {var} (x⁺)'},
                            {'Column': f'{var}_neg', 'Description': f'Negative partial sum of {var} (x⁻)'},
                            {'Column': f'{var}_delta', 'Description': f'First difference of {var} (Δx)'},
                            {'Column': f'{var}_delta_pos', 'Description': f'Positive changes in {var}'},
                            {'Column': f'{var}_delta_neg', 'Description': f'Negative changes in {var}'}
                        ])
                    
                    col_desc_df = pd.DataFrame(col_desc)
                    st.dataframe(col_desc_df, use_container_width=True)
                    
                    # Download button
                    st.markdown("---")
                    excel_data = to_excel(decomposed_data)
                    st.download_button(
                        label="📥 Download Decomposed Data (Excel)",
                        data=excel_data,
                        file_name="nardl_decomposed_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    
                    # Additional statistics download
                    st.markdown("##### Download Statistics")
                    
                    # Combine all statistics
                    all_stats = pd.concat([
                        stats_df_orig.add_suffix('_original'),
                        stats_df_pos.add_suffix('_positive'),
                        stats_df_neg.add_suffix('_negative'),
                        stats_df_delta.add_suffix('_delta')
                    ], axis=1)
                    
                    stats_excel = to_excel(all_stats)
                    st.download_button(
                        label="📥 Download Descriptive Statistics (Excel)",
                        data=stats_excel,
                        file_name="nardl_descriptive_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            else:
                st.info("👆 Please select at least one variable to decompose.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            
            1. **Upload** your Excel file (.xlsx) using the sidebar
            2. **Select** variables to decompose into positive and negative partial sums
            3. **Choose** non-decomposed variables to include in the output
            4. **Analyze** decomposition results with interactive visualizations
            5. **Test** stationarity using ADF, Phillips-Perron, and KPSS unit root tests
            6. **Download** your decomposed data for further analysis
            
            ---
            
            ### 📖 What is NARDL?
            
            The **Nonlinear Autoregressive Distributed Lag (NARDL)** model, developed by 
            Shin, Yu & Greenwood-Nimmo (2014), allows for the modeling of asymmetric 
            short-run and long-run relationships between variables.
            
            The key innovation is the **partial sum decomposition**:
            - **x⁺**: Cumulative sum of positive changes (increases)
            - **x⁻**: Cumulative sum of negative changes (decreases)
            
            This decomposition enables researchers to test whether positive and negative 
            shocks have different effects on the dependent variable.
            
            ---
            
            ### 🧪 Unit Root Tests
            
            | Test | H₀ (Null) | H₁ (Alternative) | Lag Options |
            |------|-----------|------------------|-------------|
            | **ADF** | Unit root | Stationary | Max lag, Auto-selection (AIC/BIC/t-stat) |
            | **PP** | Unit root | Stationary | Newey-West bandwidth (auto or manual) |
            | **KPSS** | Stationary | Unit root | Schwert formula or legacy |
            
            **Phillips-Perron** is preferred when there is heteroskedasticity or serial correlation in the errors.
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Features
            
            ✅ Excel file upload (.xlsx)
            
            ✅ Variable decomposition
            
            ✅ Interactive Plotly visualizations
            
            ✅ Descriptive statistics
            
            ✅ **ADF Test** with lag options
            
            ✅ **Phillips-Perron Test** 
            
            ✅ **KPSS Test**
            
            ✅ Export to Excel
            
            ---
            
            ### 🔧 Lag Options Available
            
            **ADF Test:**
            - Max lag (1-50)
            - Autolag: AIC, BIC, t-stat
            
            **Phillips-Perron:**
            - Auto Newey-West bandwidth
            - Manual lag selection (0-50)
            - Test type: Z-tau or Z-rho
            
            **KPSS:**
            - Auto (Schwert formula)
            - Legacy method
            
            ---
            
            ### 📚 Reference
            
            Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). 
            *Modelling Asymmetric Cointegration and Dynamic Multipliers 
            in a Nonlinear ARDL Framework.*
            
            ---
            
            ### ⚠️ Requirements
            
            For Phillips-Perron test:
            ```
            pip install arch
            ```
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6C757D; font-size: 0.9rem;">
        NARDL Variable Decomposition Tool | Based on Shin, Yu & Greenwood-Nimmo (2014)<br>
        Unit Root Tests: ADF, Phillips-Perron, KPSS | With Full Lag Options
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
