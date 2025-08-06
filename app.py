import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide", page_title="📱 Phân tích & Dự đoán Giá Điện Thoại")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("phones_vn.csv", encoding="utf-8-sig")

df = load_data()

# Sidebar filters
st.sidebar.header("Bộ lọc")
brands = st.sidebar.multiselect("Chọn hãng", options=sorted(df['Hãng'].unique()), default=sorted(df['Hãng'].unique()))
year_range = st.sidebar.slider("Năm ra mắt", int(df['Năm ra mắt'].min()), int(df['Năm ra mắt'].max()), 
                                (int(df['Năm ra mắt'].min()), int(df['Năm ra mắt'].max())))
ram_range = st.sidebar.slider("RAM (GB)", int(df['RAM (GB)'].min()), int(df['RAM (GB)'].max()), 
                               (int(df['RAM (GB)'].min()), int(df['RAM (GB)'].max())))
price_range = st.sidebar.slider("Giá (VNĐ)", int(df['Giá (VNĐ)'].min()), int(df['Giá (VNĐ)'].max()), 
                                 (int(df['Giá (VNĐ)'].min()), int(df['Giá (VNĐ)'].max())))

os_choices = st.sidebar.multiselect("Hệ điều hành", options=sorted(df['Hệ điều hành'].unique()), default=sorted(df['Hệ điều hành'].unique()))

# Apply filters
filtered = df[
    (df['Hãng'].isin(brands)) &
    (df['Năm ra mắt'].between(year_range[0], year_range[1])) &
    (df['RAM (GB)'].between(ram_range[0], ram_range[1])) &
    (df['Giá (VNĐ)'].between(price_range[0], price_range[1])) &
    (df['Hệ điều hành'].isin(os_choices))
]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Số mẫu", len(filtered))
col2.metric("Giá TB (VNĐ)", f"{filtered['Giá (VNĐ)'].mean():,.0f}")
col3.metric("RAM TB (GB)", f"{filtered['RAM (GB)'].mean():.1f}")
col4.metric("Pin TB (mAh)", f"{filtered['Pin (mAh)'].mean():,.0f}")

# Charts
st.subheader("📊 Giá theo hãng")
fig1 = px.box(filtered, x="Hãng", y="Giá (VNĐ)", color="Hãng", points="all")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📈 Xu hướng giá theo năm")
trend = filtered.groupby("Năm ra mắt")["Giá (VNĐ)"].mean().reset_index()
fig2 = px.line(trend, x="Năm ra mắt", y="Giá (VNĐ)", markers=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🔍 RAM vs Giá")
fig3 = px.scatter(filtered, x="RAM (GB)", y="Giá (VNĐ)", size="Pin (mAh)", color="Hãng", hover_data=["Model"])
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🧁 Phân bố hệ điều hành")
fig4 = px.pie(filtered, names="Hệ điều hành")
st.plotly_chart(fig4, use_container_width=True)

# Modeling
st.subheader("🤖 Mô hình dự đoán giá")
features = ["RAM (GB)", "Bộ nhớ (GB)", "Pin (mAh)", "Camera (MP)", "Màn hình (inch)", "Năm ra mắt"]
X = df[features]
y = df["Giá (VNĐ)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_choice = st.selectbox("Chọn model", ["Linear Regression", "Random Forest"])
if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.0f}")
c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
c3.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

# Prediction form
st.markdown("### 🔮 Thử dự đoán giá")
with st.form("predict_form"):
    ram = st.number_input("RAM (GB)", min_value=1, max_value=64, value=8)
    storage = st.number_input("Bộ nhớ (GB)", min_value=8, max_value=2048, value=128)
    battery = st.number_input("Pin (mAh)", min_value=500, max_value=10000, value=4000)
    camera = st.number_input("Camera (MP)", min_value=1, max_value=200, value=64)
    display = st.number_input("Màn hình (inch)", min_value=3.0, max_value=8.0, value=6.5, step=0.1)
    year = st.number_input("Năm ra mắt", min_value=2000, max_value=2030, value=2025)
    submit = st.form_submit_button("Dự đoán")
    if submit:
        input_data = np.array([[ram, storage, battery, camera, display, year]])
        pred_price = model.predict(input_data)[0]
        st.success(f"Giá dự đoán: {pred_price:,.0f} VNĐ")

# Data table
st.subheader("📄 Dữ liệu đã lọc")
st.dataframe(filtered)
