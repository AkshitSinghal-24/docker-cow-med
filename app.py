from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
from retrain import train_model_for_client

MODEL_DIR = "models"
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Cow Medicine Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    client = request.args.get('client')
    if not client:
        return jsonify({"error": "Client not specified. Pass client as query param like ?client=client1"}), 400

    model_path = os.path.join(MODEL_DIR, client, "model.pkl")
    le_path = os.path.join(MODEL_DIR, client, "label_encoder.pkl")
    feature_path = os.path.join(MODEL_DIR, client, "feature_columns.pkl")

    if not all(os.path.exists(p) for p in [model_path, le_path, feature_path]):
        return jsonify({"error": f"No trained model found for client: {client}. Please retrain first."}), 400

    try:
        # Load model and encoders
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        feature_columns = joblib.load(feature_path)

        # Read JSON input
        user_data = request.json
        if not user_data:
            return jsonify({"error": "No JSON body provided."}), 400

        # Expect diagnosis as a list
        diagnoses_list = user_data.get('diagnosis')
        if not diagnoses_list or not isinstance(diagnoses_list, list):
            return jsonify({"error": "Field 'diagnosis' (as list) is mandatory in JSON payload."}), 400

        # Prepare data rows for each diagnosis
        inputs = {
            "breed": user_data.get("breed", ""),
            "num_calvings": user_data.get("num_calvings", ""),
            "age": user_data.get("age", ""),
            "months_pregnant": user_data.get("months_pregnant", ""),
            "months_since_calving": user_data.get("months_since_calving", ""),
            "avg_lpd": user_data.get("avg_lpd", "")
        }

        rows = []
        for diag in diagnoses_list:
            row = inputs.copy()
            row["diagnosis"] = diag
            rows.append(row)

        user_df = pd.DataFrame(rows)

        # Convert numeric fields
        numeric_fields = ['num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd']
        user_df[numeric_fields] = user_df[numeric_fields].apply(pd.to_numeric, errors='coerce')

        # One-hot encode
        user_encoded = pd.get_dummies(user_df)
        for col in feature_columns:
            if col not in user_encoded.columns:
                user_encoded[col] = 0
        user_encoded = user_encoded[feature_columns]

        # Predict
        probs = model.predict_proba(user_encoded)

        # Sum probabilities for same medicine across multiple diagnoses
        summed_probs = probs.sum(axis=0)
        top_indices = summed_probs.argsort()[-5:][::-1]  # Top 5 medicines

        result = []
        for idx in top_indices:
            med = le.inverse_transform([idx])[0]
            conf = float(summed_probs[idx] * 100)
            result.append({
                "medicine": med,
                "confidence_percent": round(conf, 1)
            })

        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['GET'])
def retrain_client():
    client = request.args.get('client')
    if not client:
        return jsonify({"error": "Client not specified. Pass client as query param like ?client=client1"}), 400
    try:
        train_model_for_client(client)
        return jsonify({"status": f"Retraining complete for {client}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5020)
