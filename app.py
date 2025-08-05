# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide", page_title="Phone Price Dashboard", initial_sidebar_state="expanded")

# --- Utils / Load data ---
@st.cache_data
def load_data(path="phones.csv"):
    df = pd.read_csv(path)
    # Basic cleaning / ensure dtypes
    df = df.copy()
    numeric_cols = ["release_year","ram_gb","storage_gb","battery_mah","camera_mp","display_inches","price_usd"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

# If no data
if df.empty:
    st.error("Dữ liệu rỗng. Vui lòng đặt file phones.csv cùng thư mục với app.py")
    st.stop()

# --- Sidebar filters ---
st.sidebar.header("Bộ lọc")
brands = st.sidebar.multiselect("Chọn hãng", options=sorted(df['brand'].unique()), default=sorted(df['brand'].unique()))
min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
year_range = st.sidebar.slider("Năm ra mắt", min_year, max_year, (min_year, max_year))
ram_min, ram_max = int(df['ram_gb'].min()), int(df['ram_gb'].max())
ram_range = st.sidebar.slider("RAM (GB)", ram_min, ram_max, (ram_min, ram_max))
price_min, price_max = int(df['price_usd'].min()), int(df['price_usd'].max())
price_range = st.sidebar.slider("Giá (USD)", price_min, price_max, (price_min, price_max))

# Optional: OS filter (if exists)
if "os" in df.columns:
    os_choices = st.sidebar.multiselect("Hệ điều hành", options=sorted(df['os'].unique()), default=sorted(df['os'].unique()))
else:
    os_choices = None

# Apply filters
mask = df['brand'].isin(brands)
mask &= df['release_year'].between(year_range[0], year_range[1])
mask &= df['ram_gb'].between(ram_range[0], ram_range[1])
mask &= df['price_usd'].between(price_range[0], price_range[1])
if os_choices is not None:
    mask &= df['os'].isin(os_choices)

filtered = df[mask].reset_index(drop=True)

# --- KPIs ---
st.title("📱 Phone Price Dashboard")
st.markdown("Dashboard demo: EDA + mô hình dự đoán giá điện thoại (dataset mẫu).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Số mẫu (filtered)", len(filtered))
k2.metric("Giá trung bình (USD)", f"{filtered['price_usd'].mean():.2f}")
k3.metric("RAM trung bình (GB)", f"{filtered['ram_gb'].mean():.1f}")
k4.metric("Pin trung bình (mAh)", f"{filtered['battery_mah'].mean():.0f}")

st.markdown("---")

# --- Charts layout ---
left, right = st.columns([2,1])

with left:
    st.subheader("Giá theo hãng (Boxplot)")
    fig_box = px.box(filtered, x="brand", y="price_usd", points="all", color="brand",
                     labels={"price_usd":"Giá (USD)"})
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Xu hướng giá trung bình theo năm")
    trend = filtered.groupby("release_year", as_index=False)["price_usd"].mean().sort_values("release_year")
    fig_line = px.line(trend, x="release_year", y="price_usd", markers=True,
                       labels={"release_year":"Năm", "price_usd":"Giá TB (USD)"})
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("RAM vs Giá (scatter)")
    fig_scatter = px.scatter(filtered, x="ram_gb", y="price_usd", color="brand", size="battery_mah",
                             hover_data=["model"], labels={"ram_gb":"RAM (GB)","price_usd":"Giá (USD)"})
    st.plotly_chart(fig_scatter, use_container_width=True)

with right:
    st.subheader("Phân bố Hệ điều hành")
    if "os" in filtered.columns:
        os_counts = filtered['os'].value_counts().reset_index()
        os_counts.columns = ["os","count"]
        fig_pie = px.pie(os_counts, names="os", values="count", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Không có cột 'os' trong dataset.")

    st.subheader("Heatmap: RAM vs Storage (Giá TB)")
    if {"ram_gb","storage_gb","price_usd"}.issubset(filtered.columns):
        heat = filtered.groupby(["ram_gb","storage_gb"], as_index=False)["price_usd"].mean()
        heat_pivot = heat.pivot(index="ram_gb", columns="storage_gb", values="price_usd")
        st.dataframe(heat_pivot.fillna("-"))
    else:
        st.info("Thiếu cột cần thiết cho heatmap.")

st.markdown("---")

# --- Modeling: Linear Regression + RandomForest ---
st.subheader("Mô hình hồi quy: Dự đoán giá (price_usd)")

features = ["ram_gb","storage_gb","battery_mah","camera_mp","display_inches","release_year"]
for f in features:
    if f not in df.columns:
        st.error(f"Thiếu cột '{f}' trong dataset. Mô hình yêu cầu các cột: {features}")
        st.stop()

X = filtered[features].fillna(0)
y = filtered["price_usd"].fillna(0)

# If too few rows, warn
if len(filtered) < 5:
    st.warning("Số mẫu quá ít để huấn luyện mô hình tin cậy (cần >= 5 mẫu). Vẫn sẽ chạy demo trên dữ liệu hiện tại.")

test_size = st.slider("Tỉ lệ test size", 0.1, 0.5, 0.2, 0.05)
seed = st.number_input("Random seed", 0, 9999, 42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

model_choice = st.selectbox("Chọn model", options=["LinearRegression","RandomForest"])

if model_choice == "LinearRegression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=200, random_state=seed)

if st.button("Huấn luyện model"):
    with st.spinner("Đang huấn luyện..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

    st.success("Hoàn tất huấn luyện")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R²", f"{r2:.2f}")

    st.subheader("Biểu đồ: Predicted vs Actual")
    res_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    fig_pred = px.scatter(res_df, x="actual", y="predicted", trendline="ols", labels={"actual":"Actual","predicted":"Predicted"})
    st.plotly_chart(fig_pred, use_container_width=True)

    if model_choice == "RandomForest":
        importances = model.feature_importances_
        fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
        st.subheader("Feature importance")
        st.table(fi)

    # Keep trained model in session for prediction form
    st.session_state["trained_model"] = model

# --- Predict form (xài model vừa huấn luyện) ---
st.markdown("### Dự đoán giá (sử dụng model đã huấn luyện)")
if "trained_model" not in st.session_state:
    st.info("Chưa có model đã huấn luyện. Vui lòng nhấn 'Huấn luyện model' ở trên để có model dùng cho dự đoán.")

with st.form("predict_form"):
    ram = st.number_input("RAM (GB)", min_value=1, max_value=64, value=8)
    storage = st.number_input("Storage (GB)", min_value=8, max_value=2048, value=128)
    battery = st.number_input("Pin (mAh)", min_value=500, max_value=10000, value=4000)
    camera = st.number_input("Camera (MP)", min_value=1, max_value=200, value=64)
    display = st.number_input("Màn hình (inch)", min_value=3.0, max_value=8.0, value=6.5, step=0.1)
    year = st.number_input("Năm ra mắt", min_value=2000, max_value=2030, value=2021)
    submit = st.form_submit_button("Dự đoán")
    if submit:
        if "trained_model" in st.session_state:
            m = st.session_state["trained_model"]
            inp = np.array([[ram, storage, battery, camera, display, year]])
            pred = m.predict(inp)[0]
            st.success(f"Giá dự đoán: {pred:.2f} USD")
        else:
            st.error("Chưa có model. Vui lòng huấn luyện trước.")

# --- Data table & download ---
st.markdown("---")
st.subheader("Bảng dữ liệu (Filtered)")
st.dataframe(filtered.reset_index(drop=True))

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered)
st.download_button("📥 Tải CSV (filtered)", data=csv, file_name="phones_filtered.csv", mime="text/csv")
