"""
K-Means Clustering Web Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.main .block-container {
    background-color: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.main-header h1 { margin: 0; font-size: 2.5rem; font-weight: 700; }
.main-header p { margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9; }
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card h3 { margin: 0; color: #667eea; font-size: 1rem; }
.metric-card p { margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #333; }
.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 2rem 0;
}
.result-card .cluster-number { font-size: 4rem; font-weight: 900; margin: 1rem 0; }
.info-box {
    background: #e3f2fd;
    border-left: 5px solid #2196f3;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        return None, None, None


# Main header
st.markdown("""
<div class="main-header">
    <h1>🔮 K-Means Clustering App</h1>
    <p>Interactive Machine Learning Prediction System</p>
</div>
""", unsafe_allow_html=True)

model, scaler, feature_names = load_models()

if model is None:
    st.error("❌ ไม่พบไฟล์ model ในโฟลเดอร์ `models/`")
    st.info("""
    **วิธีแก้ไข:** สร้างโฟลเดอร์ `models/` แล้วใส่ไฟล์:
    - `kmeans_model.pkl`
    - `scaler.pkl`
    - `feature_names.pkl`
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 📋 About")
    st.info("""
    แอปนี้ใช้ K-Means clustering ทำนาย cluster จาก features ที่ป้อนเข้า
    
    **Model Details:**
    - Algorithm: K-Means
    - Dataset: Iris
    - Features: 4
    """)
    st.markdown("---")
    st.markdown("## 🎯 วิธีใช้งาน")
    st.markdown("""
    1. **Manual Input**: ป้อนค่าผ่าน slider
    2. **CSV Upload**: อัปโหลดไฟล์ CSV
    3. **View Results**: ดูผลทำนายและ visualization
    """)
    st.markdown("---")
    if st.button("🔄 Reset All"):
        st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Manual Prediction", "📁 Batch Prediction", "📊 Model Info"])

# ===== Tab 1: Manual Prediction =====
with tab1:
    st.markdown("### 🎯 ป้อนค่า Features")
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.5, 0.1)

    if st.button("🔮 Predict Cluster", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        cluster = int(model.predict(input_scaled)[0])
        distances = np.linalg.norm(model.cluster_centers_ - input_scaled, axis=1)
        closest_distance = distances[cluster]

        st.markdown(f"""
        <div class="result-card">
            <h2>🎉 ผลทำนาย</h2>
            <div class="cluster-number">Cluster {cluster}</div>
            <p>Distance to center: {closest_distance:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Input Features", f"{len(input_data[0])} values")
        c2.metric("Cluster", cluster)
        c3.metric("Confidence", f"{1/(1+closest_distance):.2%}")

        st.markdown("### 📏 ระยะทางไปยังแต่ละ Cluster")
        distance_df = pd.DataFrame({
            'Cluster': [f"Cluster {i}" for i in range(len(distances))],
            'Distance': distances,
            'Closest': ['✅' if i == cluster else '' for i in range(len(distances))]
        })
        st.dataframe(distance_df, use_container_width=True, hide_index=True)

        st.markdown("### 📊 Radar Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=input_scaled[0].tolist() + [input_scaled[0][0]],
            theta=feature_names + [feature_names[0]],
            fill='toself', name='Input Sample'
        ))
        fig.add_trace(go.Scatterpolar(
            r=model.cluster_centers_[cluster].tolist() + [model.cluster_centers_[cluster][0]],
            theta=feature_names + [feature_names[0]],
            fill='toself', name=f'Cluster {cluster} Center'
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ===== Tab 2: Batch Prediction =====
with tab2:
    st.markdown("### 📁 อัปโหลด CSV")
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            required_cols = set(feature_names)
            if required_cols.issubset(set(df.columns)):
                X_scaled = scaler.transform(df[feature_names].values)
                df['Predicted_Cluster'] = model.predict(X_scaled)
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Download", df.to_csv(index=False),
                                   "predictions.csv", "text/csv")
                counts = df['Predicted_Cluster'].value_counts().sort_index()
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.pie(values=counts.values,
                    names=[f"Cluster {i}" for i in counts.index],
                    title="Cluster Distribution"), use_container_width=True)
                c2.plotly_chart(px.scatter(df, x=feature_names[0], y=feature_names[1],
                    color='Predicted_Cluster', title="Feature Space"),
                    use_container_width=True)
            else:
                st.error(f"❌ คอลัมน์ไม่ตรง! ต้องการ: {required_cols}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===== Tab 3: Model Info =====
with tab3:
    c1, c2 = st.columns(2)
    c1.markdown(f"""
    #### 🔧 Parameters
    - Algorithm: K-Means
    - Clusters: {model.n_clusters}
    - Max Iter: {model.max_iter}
    - Inertia: {model.inertia_:.4f}
    """)
    c2.markdown(f"""
    #### 📊 Features
    - จำนวน: {len(feature_names)}
    - ชื่อ: {', '.join(feature_names)}
    """)
    st.markdown("### 📍 Cluster Centers")
    centers_df = pd.DataFrame(model.cluster_centers_, columns=feature_names,
        index=[f"Cluster {i}" for i in range(model.n_clusters)])
    st.dataframe(centers_df, use_container_width=True)
    st.plotly_chart(px.imshow(model.cluster_centers_, x=feature_names,
        y=[f"Cluster {i}" for i in range(model.n_clusters)],
        color_continuous_scale='Viridis', aspect='auto',
        title="Cluster Centers Heatmap"), use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>🎓 ML Course | Built with ❤️ Streamlit © 2026</div>",
            unsafe_allow_html=True)