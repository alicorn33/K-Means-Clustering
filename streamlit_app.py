"""
K-Means Clustering Web Application
Author: Machine Learning Course
Description: Interactive web app for K-Means clustering predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Container styling */
    .main .block-container {
        background-color: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        margin: 0;
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    
    .result-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .result-card .cluster-number {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load trained model and scaler"""
    try:
        model = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure the models folder exists with required files.")
        return None, None, None

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔮 K-Means Clustering App</h1>
    <p>Interactive Machine Learning Prediction System</p>
</div>
""", unsafe_allow_html=True)

# Load models
model, scaler, feature_names = load_models()

if model is not None:
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 About")
        st.info("""
        This application uses a trained K-Means clustering model to predict 
        cluster assignments based on input features.
        
        **Model Details:**
        - Algorithm: K-Means
        - Dataset: Iris
        - Features: 4
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 How to Use")
        st.markdown("""
        1. **Manual Input**: Enter feature values using sliders
        2. **CSV Upload**: Upload a CSV file for batch predictions
        3. **View Results**: See cluster assignments and visualizations
        """)
        
        st.markdown("---")
        if st.button("🔄 Reset All"):
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Manual Prediction", "📁 Batch Prediction", "📊 Model Information"])
    
    # Tab 1: Manual Prediction
    with tab1:
        st.markdown("### 🎯 Enter Feature Values")
        
        # Create input columns
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0,
                max_value=8.0,
                value=5.5,
                step=0.1,
                help="Enter sepal length in centimeters"
            )
            
            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Enter sepal width in centimeters"
            )
        
        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                value=4.0,
                step=0.1,
                help="Enter petal length in centimeters"
            )
            
            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Enter petal width in centimeters"
            )
        
        # Predict button
        if st.button("🔮 Predict Cluster", use_container_width=True):
            # Prepare input data
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Scale input data
            input_scaled = scaler.transform(input_data)
            
            # Predict cluster
            cluster = model.predict(input_scaled)[0]
            
            # Calculate distances to all cluster centers
            distances = np.linalg.norm(model.cluster_centers_ - input_scaled, axis=1)
            closest_distance = distances[cluster]
            
            # Display results
            st.markdown("---")
            st.markdown(f"""
            <div class="result-card">
                <h2>🎉 Prediction Result</h2>
                <div class="cluster-number">Cluster {cluster}</div>
                <p>Distance to cluster center: {closest_distance:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Input Features</h3>
                    <p style="font-size: 1rem;">{len(input_data[0])} values</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Cluster Assigned</h3>
                    <p>{cluster}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence</h3>
                    <p style="font-size: 1.5rem;">{1/(1+closest_distance):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display distances to all clusters
            st.markdown("### 📏 Distances to All Cluster Centers")
            distance_df = pd.DataFrame({
                'Cluster': [f"Cluster {i}" for i in range(len(distances))],
                'Distance': distances,
                'Closest': ['✅' if i == cluster else '' for i in range(len(distances))]
            })
            st.dataframe(distance_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("### 📊 Feature Visualization")
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=input_scaled[0].tolist() + [input_scaled[0][0]],
                theta=feature_names + [feature_names[0]],
                fill='toself',
                name='Input Sample',
                line_color='rgb(102, 126, 234)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=model.cluster_centers_[cluster].tolist() + [model.cluster_centers_[cluster][0]],
                theta=feature_names + [feature_names[0]],
                fill='toself',
                name=f'Cluster {cluster} Center',
                line_color='rgb(255, 99, 132)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True),
                ),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.markdown("### 📁 Upload CSV File")
        st.info("Upload a CSV file with the same feature columns for batch predictions.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.markdown("### 📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check if columns match
                required_cols = set(feature_names)
                actual_cols = set(df.columns)
                
                if required_cols.issubset(actual_cols):
                    # Prepare data
                    X_batch = df[feature_names].values
                    
                    # Scale data
                    X_batch_scaled = scaler.transform(X_batch)
                    
                    # Predict
                    predictions = model.predict(X_batch_scaled)
                    
                    # Add predictions to dataframe
                    df['Predicted_Cluster'] = predictions
                    
                    # Display results
                    st.markdown("### ✅ Predictions Complete")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    # Cluster distribution
                    st.markdown("### 📊 Cluster Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cluster_counts = df['Predicted_Cluster'].value_counts().sort_index()
                        fig_pie = px.pie(
                            values=cluster_counts.values,
                            names=[f"Cluster {i}" for i in cluster_counts.index],
                            title="Cluster Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        fig_bar = px.bar(
                            x=[f"Cluster {i}" for i in cluster_counts.index],
                            y=cluster_counts.values,
                            title="Samples per Cluster",
                            labels={'x': 'Cluster', 'y': 'Count'},
                            color=cluster_counts.values,
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # 2D visualization
                    st.markdown("### 🎨 2D Feature Space Visualization")
                    fig_2d = px.scatter(
                        df,
                        x=feature_names[0],
                        y=feature_names[1],
                        color='Predicted_Cluster',
                        title=f"{feature_names[0]} vs {feature_names[1]}",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                else:
                    st.error(f"""
                    ❌ **Column mismatch!**
                    
                    Required columns: {', '.join(required_cols)}
                    
                    Found columns: {', '.join(actual_cols)}
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Tab 3: Model Information
    with tab3:
        st.markdown("### 📋 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Model Parameters")
            st.markdown(f"""
            - **Algorithm**: K-Means
            - **Number of Clusters**: {model.n_clusters}
            - **Max Iterations**: {model.max_iter}
            - **Random State**: {model.random_state}
            - **N Init**: {model.n_init}
            """)
        
        with col2:
            st.markdown("#### 📊 Model Statistics")
            st.markdown(f"""
            - **Inertia**: {model.inertia_:.4f}
            - **Number of Features**: {len(feature_names)}
            - **Feature Names**: {', '.join(feature_names)}
            """)
        
        st.markdown("### 📍 Cluster Centers")
        centers_df = pd.DataFrame(
            model.cluster_centers_,
            columns=feature_names,
            index=[f"Cluster {i}" for i in range(model.n_clusters)]
        )
        st.dataframe(centers_df, use_container_width=True)
        
        st.markdown("### 📈 Cluster Center Visualization")
        fig_centers = px.imshow(
            model.cluster_centers_,
            labels=dict(x="Features", y="Clusters", color="Value"),
            x=feature_names,
            y=[f"Cluster {i}" for i in range(model.n_clusters)],
            color_continuous_scale='Viridis',
            aspect='auto'
        )
        st.plotly_chart(fig_centers, use_container_width=True)
        
        st.markdown("### 💡 How K-Means Works")
        st.markdown("""
        <div class="info-box">
        <strong>K-Means Clustering Algorithm:</strong>
        <ol>
            <li><strong>Initialization:</strong> Randomly place K cluster centers in the feature space</li>
            <li><strong>Assignment:</strong> Assign each data point to the nearest cluster center</li>
            <li><strong>Update:</strong> Recalculate cluster centers as the mean of assigned points</li>
            <li><strong>Iteration:</strong> Repeat steps 2-3 until convergence</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎓 <strong>Machine Learning for Python Programming Course</strong></p>
    <p>Built with ❤️ using Streamlit | © 2026</p>
</div>
""", unsafe_allow_html=True)