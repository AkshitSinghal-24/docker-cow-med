import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = "client_data"
MODEL_DIR = "models"

def train_model_for_client(client_name):
    client_path = os.path.join(DATA_DIR, client_name)
    excel_path = os.path.join(client_path, "cow_data.xlsx")

    if not os.path.exists(excel_path):
        print(f"❌ Excel file missing for {client_name}")
        return

    print(f"✅ Training model for {client_name}...")

    try:
        obs_df = pd.read_excel(excel_path, sheet_name="Observation to Diagnosis")
        rx_df = pd.read_excel(excel_path, sheet_name="Diagnosis to Rx")

        # Filter bad rows
        obs_df = obs_df[~obs_df['care_calendar_id'].astype(str).str.contains("care_calendar_id", na=False)]

        # Pivot and preprocess
        pivot = obs_df.pivot_table(index='care_calendar_id', columns='Observation', values='Response', aggfunc='first').reset_index()
        diagnoses = obs_df[['care_calendar_id', 'Diagnosis']].drop_duplicates()
        merged_obs = pd.merge(pivot, diagnoses, on='care_calendar_id', how='left')

        # Merge with Rx
        full_data = pd.merge(merged_obs, rx_df, on='Diagnosis', how='inner')

        # Rename columns
        full_data = full_data.rename(columns={
            'Diagnosis': 'diagnosis',
            'Breed': 'breed',
            'No. of Calvings': 'num_calvings',
            'Age as of 24-Feb-2025': 'age',
            'Months Pregnant as of 24-Feb-2025': 'months_pregnant',
            'Months Since Calving As of 24-Feb-2025': 'months_since_calving',
            'Average Animal LPD': 'avg_lpd'
        })

        # Drop rows with missing important fields
        required_cols = ['diagnosis', 'breed', 'num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd', 'Medicine']
        full_data = full_data.dropna(subset=required_cols)

        # Label encode target
        le = LabelEncoder()
        full_data['Medicine_enc'] = le.fit_transform(full_data['Medicine'])

        # One-hot encode features
        X = pd.get_dummies(full_data[['diagnosis', 'breed', 'num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd']])
        y = full_data['Medicine_enc']

        # Train
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
        model.fit(X, y)

        # Save model and encoder
        save_path = os.path.join(MODEL_DIR, client_name)
        os.makedirs(save_path, exist_ok=True)

        joblib.dump(model, os.path.join(save_path, "model.pkl"))
        joblib.dump(le, os.path.join(save_path, "label_encoder.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(save_path, "feature_columns.pkl"))

        print(f"✅ Model saved for {client_name} at {save_path}")

    except Exception as e:
        print(f"❌ Failed for {client_name}: {e}")

def retrain_all_clients():
    clients = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for client in clients:
        train_model_for_client(client)

if __name__ == "__main__":
    retrain_all_clients()
