import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide", page_title="Phone Price Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("phones.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Bộ lọc")
brands = st.sidebar.multiselect("Chọn hãng", options=df['brand'].unique(), default=df['brand'].unique())
min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
year_range = st.sidebar.slider("Năm ra mắt", min_year, max_year, (min_year, max_year))
ram_range = st.sidebar.slider("RAM (GB)", int(df['ram_gb'].min()), int(df['ram_gb'].max()), 
                               (int(df['ram_gb'].min()), int(df['ram_gb'].max())))
price_range = st.sidebar.slider("Giá (USD)", int(df['price_usd'].min()), int(df['price_usd'].max()),
                                 (int(df['price_usd'].min()), int(df['price_usd'].max())))

filtered = df[
    (df['brand'].isin(brands)) &
    (df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1]) &
    (df['ram_gb'] >= ram_range[0]) & (df['ram_gb'] <= ram_range[1]) &
    (df['price_usd'] >= price_range[0]) & (df['price_usd'] <= price_range[1])
]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Số mẫu", len(filtered))
col2.metric("Giá TB (USD)", f"{filtered['price_usd'].mean():.2f}")
col3.metric("RAM TB (GB)", f"{filtered['ram_gb'].mean():.2f}")
col4.metric("Pin TB (mAh)", f"{filtered['battery_mah'].mean():.0f}")

# Biểu đồ
st.subheader("Giá theo hãng")
fig1 = px.box(filtered, x="brand", y="price_usd", points="all", color="brand")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Xu hướng giá theo năm")
fig2 = px.line(filtered.groupby("release_year")["price_usd"].mean().reset_index(),
               x="release_year", y="price_usd")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("RAM vs Giá")
fig3 = px.scatter(filtered, x="ram_gb", y="price_usd", color="brand", size="battery_mah",
                  hover_data=["model"])
st.plotly_chart(fig3, use_container_width=True)

# Mô hình dự đoán
st.subheader("Mô hình hồi quy dự đoán giá")
X = df[["ram_gb", "storage_gb", "battery_mah", "camera_mp", "display_inches", "release_year"]]
y = df["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
col3.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

# Form dự đoán
st.markdown("### Thử dự đoán giá")
with st.form("predict_form"):
    ram = st.number_input("RAM (GB)", min_value=1, max_value=16, value=8)
    storage = st.number_input("Storage (GB)", min_value=8, max_value=512, value=128)
    battery = st.number_input("Pin (mAh)", min_value=1000, max_value=6000, value=4000)
    camera = st.number_input("Camera (MP)", min_value=5, max_value=200, value=64)
    display = st.number_input("Màn hình (inch)", min_value=4.0, max_value=7.5, value=6.5)
    year = st.number_input("Năm ra mắt", min_value=2015, max_value=2025, value=2021)
    submitted = st.form_submit_button("Dự đoán")
    if submitted:
        input_data = np.array([[ram, storage, battery, camera, display, year]])
        pred_price = model.predict(input_data)[0]
        st.success(f"Giá dự đoán: {pred_price:.2f} USD")

# Bảng dữ liệu
st.subheader("Bảng dữ liệu")
st.dataframe(filtered)
