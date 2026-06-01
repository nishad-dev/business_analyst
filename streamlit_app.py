# app.py - Clean version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
from datetime import datetime
from io import BytesIO

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Analytics", layout="wide")

st.title("📊 Enterprise Sales Analytics Platform")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
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

if st.session_state.data_loaded:
    df = st.session_state.df
    
    st.subheader("Column Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_col = st.selectbox("Date Column", [''] + list(df.columns))
        sales_col = st.selectbox("Sales Column", [''] + list(df.columns))
        product_col = st.selectbox("Product Column", [''] + list(df.columns))
    
    with col2:
        region_col = st.selectbox("Region Column", [''] + list(df.columns))
        quantity_col = st.selectbox("Quantity Column", [''] + list(df.columns))
        price_col = st.selectbox("Price Column", [''] + list(df.columns))
    
    with st.expander("Data Preview"):
        st.dataframe(df.head(10))
    
    if st.button("Run Analysis", type="primary"):
        if not sales_col:
            st.error("Please select a sales column")
        else:
            with st.spinner("Processing..."):
                df_clean = df.copy()
                
                if date_col:
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                    df_clean = df_clean.dropna(subset=[date_col])
                    df_clean['Month'] = df_clean[date_col].dt.month
                    df_clean['Year'] = df_clean[date_col].dt.year
                
                if sales_col:
                    df_clean[sales_col] = pd.to_numeric(df_clean[sales_col], errors='coerce')
                    df_clean = df_clean[df_clean[sales_col] > 0]
                
                st.session_state.df_clean = df_clean
                st.session_state.sales_col = sales_col
                st.session_state.product_col = product_col
                st.session_state.region_col = region_col
                st.session_state.date_col = date_col
                st.session_state.analysis_complete = True
                st.rerun()

if st.session_state.get('analysis_complete', False):
    df = st.session_state.df_clean
    sales_col = st.session_state.sales_col
    product_col = st.session_state.product_col
    region_col = st.session_state.region_col
    date_col = st.session_state.date_col
    
    # KPIs
    total_revenue = df[sales_col].sum()
    avg_transaction = df[sales_col].mean()
    
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Avg Transaction", f"${avg_transaction:,.2f}")
    col3.metric("Transactions", len(df))
    if product_col:
        col4.metric("Products", df[product_col].nunique())
    
    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    filtered_df = df.copy()
    
    with col1:
        if region_col:
            regions = ['All'] + sorted(df[region_col].unique().tolist())
            selected_region = st.selectbox("Region", regions)
            if selected_region != 'All':
                filtered_df = filtered_df[filtered_df[region_col] == selected_region]
    
    with col2:
        if product_col:
            products = ['All'] + sorted(df[product_col].unique().tolist())
            selected_product = st.selectbox("Product", products)
            if selected_product != 'All':
                filtered_df = filtered_df[filtered_df[product_col] == selected_product]
    
    # Visualizations
    st.subheader("Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Top Products", "Regional Sales", "Monthly Trends"])
    
    with tab1:
        if product_col:
            top = filtered_df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=top.values, y=top.index, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if region_col:
            region = filtered_df.groupby(region_col)[sales_col].sum()
            fig = px.pie(values=region.values, names=region.index, hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if date_col and 'Month' in filtered_df.columns:
            monthly = filtered_df.groupby('Month')[sales_col].sum()
            fig = px.line(x=monthly.index, y=monthly.values, markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.subheader("Export")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Report"):
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "report.csv", "text/csv")
    
    with col2:
        if st.button("Download Summary"):
            summary = pd.DataFrame({
                'Metric': ['Revenue', 'Avg Transaction', 'Transactions'],
                'Value': [f'${total_revenue:,.2f}', f'${avg_transaction:,.2f}', len(df)]
            })
            csv = summary.to_csv(index=False)
            st.download_button("Download Summary", csv, "summary.csv", "text/csv")

st.markdown("---")
st.markdown("Powered by Machine Learning")
