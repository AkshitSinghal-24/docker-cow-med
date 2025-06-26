# 🐄 Cow Medicine Predictor API 🚀

> Built with ☕️, late nights, and funded AI labor 😤🤖...  
> By **Akshit Singhal** (with ChatGPT playing unpaid intern 🤖)

---

## 📋 Project Summary

A multi-client XGBoost-powered Flask API to predict recommended cow medicines based on diagnosis and optional cow data like breed, age, calving info, etc.

Features:

✅ Per-client model training  
✅ Supports daily retraining as new rows get added  
✅ JSON-based REST API  
✅ Docker-friendly  
✅ Designed for deployment or local testing  

---

## 📂 Folder Structure

```
cow_medicine_predictor/
├── client_data/
│   ├── client_A/
│   │   └── cow_data.xlsx
│   └── client_B/
│       └── cow_data.xlsx
├── models/
│   └── client_A/
│       └── model.pkl, etc.
├── retrain.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ✅ Setup Instructions (Local)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, create one:

```
flask
pandas
scikit-learn
xgboost
openpyxl
joblib
```

---

### 2. Train Models Initially (for each client)

```bash
python retrain.py
```

This will:

- Read Excel data from each client folder  
- Train an XGBoost model  
- Save model files (`model.pkl`, `label_encoder.pkl`, etc.) under `/models/{client}/`

---

### 3. Run Flask API Locally

```bash
python app.py
```

API will run at:

```
http://127.0.0.1:5020/
```

---

## ✅ API Endpoints

---

### ✅ 1. Health Check

```bash
curl http://127.0.0.1:5020/
```

Returns:

```
Cow Medicine API is running.
```

---

### ✅ 2. Manual Retrain for a Client

**GET** `/retrain`

**Example:**

```bash
curl http://127.0.0.1:5020/retrain?client=client_A
```

Response:

```json
{
  "status": "Retraining complete for client_A"
}
```

---

### ✅ 3. Predict Medicines

**POST** `/predict?client=client_A`

✅ **Mandatory:**

- Query Param: `client`
- JSON Body Field: `diagnosis`
- "diagnosis" (List of strings like ["Mastitis", "Metritis"])

**Example cURL Request:**

```bash
curl -X POST http://127.0.0.1:5020/predict?client=client_A \
  -H "Content-Type: application/json" \
  -d '{
  "diagnosis": ["Mastitis", "Metritis"],
  "breed": "Holstein",
  "num_calvings": 2,
  "age": 5,
  "months_pregnant": 3,
  "months_since_calving": 2,
  "avg_lpd": 15.0
}'
```

✅ **Minimum Working Example:**

```bash
curl -X POST http://127.0.0.1:5020/predict?client=client_A \
  -H "Content-Type: application/json" \
  -d '{"diagnosis": ["Mastitis"]}'
```

✅ **Response Format:**

```{
  "predictions": [
    {"medicine": "PREDNISOLONE ACETATE INJ 10 ML", "confidence_percent": 38.1},
    {"medicine": "CADISTIN INJ 100 ML", "confidence_percent": 32.3},
    {"medicine": "ATN-MB INJ 100 ML", "confidence_percent": 9.6},
    {"medicine": "25D 500ML", "confidence_percent": 6.3}
  ]
}
```

---

## ✅ 4. Automating Daily Retraining (Cron Job)

For Linux/Mac servers:

```bash
crontab -e
```

Then add:

```
0 12 * * * cd /path/to/cow_medicine_predictor && /path/to/python retrain.py
```

This retrains all client models **daily at 12 PM**.

---

## ✅ 5. Running with Docker 🐳

### Example Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5020

CMD ["python", "app.py"]
```

---

### ✅ Build Docker Image:

```bash
docker build -t cow_medicine_api .
```

---

### ✅ Run Docker Container:

```bash
docker run -d -p 5020:5020 cow_medicine_api
```

This will expose the API at:

```
http://localhost:5020/
```

---

### ✅ Live Testing via Docker:

```bash
curl http://localhost:5020/
```

Or hit the `/predict` or `/retrain` endpoints as shown earlier.

---

## ✅ 6. Adding New Clients

- Place their `cow_data.xlsx` inside:

```
client_data/client_C/cow_data.xlsx
```

- Then retrain:

```bash
python retrain.py
```

New client will be ready to use.

---

## ✅ Author:

Akshit Singhal 😎  
*(With ChatGPT doing exactly what it was paid for 💸)*  

---

