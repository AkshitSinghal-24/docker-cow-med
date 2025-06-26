# 🐄 Cow Medicine Predictor API 🚀

> Built with ☕️, late nights, and funded AI labor 😤🤖...  
> By **Akshit Singhal** (with ChatGPT doing what it was paid for 💸)

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

## ✅ Setup Instructions

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

Flask will start on:

```
http://127.0.0.1:5000/
```

If you want auto-reload for development:

```bash
export FLASK_ENV=development
python app.py
```

On Windows CMD:

```cmd
set FLASK_ENV=development
python app.py
```

---

## ✅ API Endpoints

---

### ✅ 1. Health Check (Optional)

If you added a `/` route:

```bash
curl http://127.0.0.1:5000/
```

---

### ✅ 2. Manual Retrain for a Client

**GET** `/retrain`

**Example:**

```bash
curl http://127.0.0.1:5000/retrain?client=client_A
```

Response:

```json
{
  "message": "Model retrained successfully for client_A"
}
```

---

### ✅ 3. Predict Medicines

**POST** `/predict`

**Example cURL Request:**

```bash
curl -X POST http://127.0.0.1:5000/predict?client=client_A \
  -H "Content-Type: application/json" \
  -d '{
        "diagnosis": "Mastitis",
        "breed": "Holstein",
        "num_calvings": 2,
        "age": 5,
        "months_pregnant": 3,
        "months_since_calving": 2,
        "avg_lpd": 15.0
      }'
```

**Response Example:**

```json
{
  "predictions": [
    {"medicine": "Oxytetracycline", "confidence_percent": 80.0},
    {"medicine": "Penicillin", "confidence_percent": 15.0}
  ]
}
```

✅ Only `diagnosis` is mandatory, rest of the fields can be left blank or null (model will handle missing).

---

## ✅ 4. Automating Daily Retraining (Cron Job)

For Linux/Mac servers with cron:

```bash
crontab -e
```

Then add:

```
0 12 * * * cd /path/to/cow_medicine_predictor && /path/to/python retrain.py
```

This will retrain all client models **daily at 12 PM**.

---

## ✅ 5. Optional: Docker Deployment

### Example Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
```

### Build Docker Image:

```bash
docker build -t cow_medicine_api .
```

### Run Docker Container:

```bash
docker run -d -p 5000:5000 cow_medicine_api
```

---

## ✅ 6. Adding New Clients

- Create a new folder:

```
client_data/client_C/cow_data.xlsx
```

- Then retrain:

```bash
python retrain.py
```

New client will now work with API.

---

## ✅ Author:

Akshit Singhal 😎  
*(With ChatGPT doing what it was paid for 💸)*  

---

