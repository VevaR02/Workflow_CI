import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#latest
mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv('dataset_preprocessing/dataset_ready.csv')

X = df.drop('status', axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest_Lokal"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Melatih model di MLflow Lokal...")
    rf.fit(X_train, y_train)
    print("Selesai! Eksperimen tersimpan di lokal.")