import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model():
    # 1. LOAD DATAS
    data_path = os.path.join('data', 'laptop_data.csv')
    df = pd.read_csv(data_path)
    
    X = df[['ram', 'ssd', 'weight', 'screen']]
    y = df['price']

    # 2. BUILD PIPELINE
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # 3. TRAIN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # 4. Save WHOLE PIPELINE AS JOBLINE
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/laptop_pipeline.joblib')
    print("✅ Model Pipeline saved to models/laptop_pipeline.joblib")

if __name__ == "__main__":
    train_model()