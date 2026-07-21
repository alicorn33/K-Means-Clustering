"""
Train K-Means Model for Iris Dataset
This script trains a K-Means clustering model and saves it for the Streamlit app
"""
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    """Train K-Means model and save to models/ folder"""
    
    print("=" * 60)
    print("🚀 Starting K-Means Model Training")
    print("=" * 60)
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    print("✅ Created/verified 'models/' directory")
    
    # Load Iris dataset
    print("\n📊 Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names
    
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Features: {', '.join(feature_names)}")
    
    # Scale features
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features scaled successfully")
    
    # Train K-Means model
    print("\n🎯 Training K-Means model...")
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X_scaled)
    print(f"✅ Model trained successfully")
    print(f"   Number of clusters: {kmeans.n_clusters}")
    print(f"   Inertia: {kmeans.inertia_:.4f}")
    print(f"   Iterations: {kmeans.n_iter_}")
    
    # Save model
    print("\n💾 Saving model...")
    model_path = 'models/kmeans_model.pkl'
    joblib.dump(kmeans, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save scaler
    print("\n💾 Saving scaler...")
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Save feature names
    print("\n💾 Saving feature names...")
    feature_names_path = 'models/feature_names.pkl'
    joblib.dump(feature_names, feature_names_path)
    print(f"✅ Feature names saved to: {feature_names_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print("\n📁 Files created in 'models/' folder:")
    print("   ├── kmeans_model.pkl")
    print("   ├── scaler.pkl")
    print("   └── feature_names.pkl")
    print("\n🚀 You can now run the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print("=" * 60)
    
    # Test prediction
    print("\n🧪 Testing prediction with sample data...")
    sample = X_scaled[0:1]  # First sample
    prediction = kmeans.predict(sample)[0]
    print(f"✅ Sample prediction: Cluster {prediction}")
    
    return kmeans, scaler, feature_names


if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\n💡 Please check:")
        print("   1. All required packages are installed")
        print("   2. You have write permissions in this directory")
        print("   3. The 'models/' folder can be created")