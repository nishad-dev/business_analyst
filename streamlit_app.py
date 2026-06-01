# streamlit_app.py - Simplified version without plotly
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
from datetime import datetime
from io import BytesIO

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Sales Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6B7280;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Header
st.markdown('<div class="main-header">Enterprise Sales Analytics Platform</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main Content
if st.session_state.data_loaded:
    df = st.session_state.df
    
    st.markdown('<div class="section-header">Column Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_col = st.selectbox("Date Column", [''] + list(df.columns))
        sales_col = st.selectbox("Sales/Revenue Column", [''] + list(df.columns))
        product_col = st.selectbox("Product Column", [''] + list(df.columns))
    
    with col2:
        region_col = st.selectbox("Region/City Column", [''] + list(df.columns))
        quantity_col = st.selectbox("Quantity Column", [''] + list(df.columns))
        price_col = st.selectbox("Price Column", [''] + list(df.columns))
    
    with st.expander("Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    if st.button("Run Analysis", type="primary", use_container_width=True):
        if not sales_col:
            st.error("Please select a sales column")
        else:
            with st.spinner("Processing data..."):
                # Data cleaning
                df_clean = df.copy()
                
                if date_col:
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                    df_clean = df_clean.dropna(subset=[date_col])
                    df_clean['Month'] = df_clean[date_col].dt.month
                    df_clean['Year'] = df_clean[date_col].dt.year
                
                if sales_col:
                    df_clean[sales_col] = pd.to_numeric(df_clean[sales_col], errors='coerce')
                    df_clean = df_clean[df_clean[sales_col] > 0]
                
                if quantity_col:
                    df_clean[quantity_col] = pd.to_numeric(df_clean[quantity_col], errors='coerce')
                
                if price_col:
                    df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors='coerce')
                
                st.session_state.df_clean = df_clean
                st.session_state.date_col = date_col
                st.session_state.sales_col = sales_col
                st.session_state.product_col = product_col
                st.session_state.region_col = region_col
                st.session_state.quantity_col = quantity_col
                st.session_state.price_col = price_col
                st.session_state.analysis_complete = True
                st.rerun()

# Analysis Results
if st.session_state.get('analysis_complete', False):
    df = st.session_state.df_clean
    date_col = st.session_state.date_col
    sales_col = st.session_state.sales_col
    product_col = st.session_state.product_col
    region_col = st.session_state.region_col
    quantity_col = st.session_state.quantity_col
    price_col = st.session_state.price_col
    
    # Calculate KPIs
    total_revenue = df[sales_col].sum()
    avg_transaction = df[sales_col].mean()
    total_transactions = len(df)
    unique_products = df[product_col].nunique() if product_col else 0
    unique_regions = df[region_col].nunique() if region_col else 0
    
    # Display KPIs
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-label">Total Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_transaction:,.0f}</div>
            <div class="metric-label">Avg Transaction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_transactions:,}</div>
            <div class="metric-label">Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if product_col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_products:,}</div>
                <div class="metric-label">Products</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        if region_col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_regions:,}</div>
                <div class="metric-label">Regions</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Filters
    st.markdown('<div class="section-header">Interactive Filters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    filtered_df = df.copy()
    
    with col1:
        if region_col:
            regions = ['All'] + sorted(df[region_col].dropna().unique().tolist())
            selected_region = st.selectbox("Region", regions)
            if selected_region != 'All':
                filtered_df = filtered_df[filtered_df[region_col] == selected_region]
    
    with col2:
        if product_col:
            products = ['All'] + sorted(df[product_col].dropna().unique().tolist())
            selected_product = st.selectbox("Product", products)
            if selected_product != 'All':
                filtered_df = filtered_df[filtered_df[product_col] == selected_product]
    
    with col3:
        if date_col and 'Year' in df.columns:
            years = ['All'] + sorted(df['Year'].dropna().unique().tolist())
            selected_year = st.selectbox("Year", years)
            if selected_year != 'All':
                filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")
    
    # Visualizations using matplotlib
    st.markdown('<div class="section-header">Analytics Dashboard</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Product Performance", "Regional Analysis", "Time Series"])
    
    with tab1:
        if product_col and len(filtered_df) > 0:
            top_products = filtered_df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_products)), top_products.values, color='#1E3A5F')
            ax.set_yticks(range(len(top_products)))
            ax.set_yticklabels(top_products.index)
            ax.set_xlabel('Revenue (USD)')
            ax.set_title('Top 10 Products by Revenue')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"🏆 Top Product: {top_products.index[0]} (${top_products.iloc[0]:,.2f})")
    
    with tab2:
        if region_col and len(filtered_df) > 0:
            region_sales = filtered_df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(region_sales.index, region_sales.values, color='#1E3A5F')
            ax.set_xlabel('Region')
            ax.set_ylabel('Revenue (USD)')
            ax.set_title('Sales by Region')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            if len(region_sales) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"Best Region: {region_sales.index[0]} (${region_sales.iloc[0]:,.2f})")
                with col2:
                    st.warning(f"Region to Improve: {region_sales.index[-1]} (${region_sales.iloc[-1]:,.2f})")
    
    with tab3:
        if date_col and 'Month' in filtered_df.columns:
            monthly_sales = filtered_df.groupby('Month')[sales_col].sum().sort_index()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8, color='#1E3A5F')
            ax.set_xlabel('Month')
            ax.set_ylabel('Revenue (USD)')
            ax.set_title('Monthly Revenue Trend')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 13))
            plt.tight_layout()
            st.pyplot(fig)
            
            peak_month = monthly_sales.idxmax()
            st.info(f"📈 Peak Sales Month: {peak_month} (${monthly_sales[peak_month]:,.2f})")
    
    # Export Section
    st.markdown('<div class="section-header">Export Reports</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Excel Report", use_container_width=True):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Filtered Data', index=False)
                if product_col:
                    summary = filtered_df.groupby(product_col)[sales_col].agg(['sum', 'mean', 'count']).round(2)
                    summary.to_excel(writer, sheet_name='Product Analysis')
                if region_col:
                    summary = filtered_df.groupby(region_col)[sales_col].agg(['sum', 'mean', 'count']).round(2)
                    summary.to_excel(writer, sheet_name='Regional Analysis')
            output.seek(0)
            st.download_button("Download", output, "sales_report.xlsx", use_container_width=True)
    
    with col2:
        if st.button("Download Summary CSV", use_container_width=True):
            summary_data = {
                'Metric': ['Total Revenue', 'Avg Transaction', 'Total Transactions'],
                'Value': [f'${total_revenue:,.2f}', f'${avg_transaction:,.2f}', total_transactions]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button("Download", csv, "summary_report.csv", "text/csv", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Powered by Machine Learning | Real-time Analytics")
