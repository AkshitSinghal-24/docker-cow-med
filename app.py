from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
from retrain import train_model_for_client

MODEL_DIR = "models"
app = Flask(__name__)


@app.route('/')
def home():
    return "Cow Medicine API is running."

@app.route('/predict', methods=['POST'])
def predict():
    client = request.args.get('client')
    if not client:
        return jsonify({"error": "Client not specified"}), 400

    client_model_path = os.path.join(MODEL_DIR, client, "model.pkl")
    client_le_path = os.path.join(MODEL_DIR, client, "label_encoder.pkl")
    client_feature_path = os.path.join(MODEL_DIR, client, "feature_columns.pkl")

    if not all(os.path.exists(p) for p in [client_model_path, client_le_path, client_feature_path]):
        return jsonify({"error": f"No trained model found for {client}"}), 400

    try:
        model = joblib.load(client_model_path)
        le = joblib.load(client_le_path)
        feature_columns = joblib.load(client_feature_path)

        user_data = request.json
        user_df = pd.DataFrame([user_data])

        # One-hot encode
        user_encoded = pd.get_dummies(user_df)
        for col in feature_columns:
            if col not in user_encoded.columns:
                user_encoded[col] = 0
        user_encoded = user_encoded[feature_columns]

        probs = model.predict_proba(user_encoded)[0]
        top_idx = probs.argsort()[-2:][::-1]
        meds = le.inverse_transform(top_idx)

        result = []
        for idx in top_idx:
            result.append({
                "medicine": le.inverse_transform([idx])[0],
                "confidence_percent": float(probs[idx] * 100)
            })

        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['GET'])
def retrain_client():
    client = request.args.get('client')
    if not client:
        return jsonify({"error": "Client not specified"}), 400
    try:
        train_model_for_client(client)
        return jsonify({"status": f"Retraining complete for {client}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5020)

