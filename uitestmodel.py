import pandas as pd
from tkinter import Tk, Label, Entry, Button, Text, Scrollbar, END, messagebox, StringVar, Listbox, MULTIPLE
from tkinter.ttk import Combobox
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === Load and preprocess data ===
excel_path = "cow_data.xlsx"

try:
    obs_df = pd.read_excel(excel_path, sheet_name="Observation to Diagnosis")
    rx_df = pd.read_excel(excel_path, sheet_name="Diagnosis to Rx")
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

obs_df = obs_df[~obs_df['care_calendar_id'].astype(str).str.contains("care_calendar_id", na=False)]

# Pivot observation data
pivot = obs_df.pivot_table(index='care_calendar_id', columns='Observation', values='Response', aggfunc='first').reset_index()
diagnoses = obs_df[['care_calendar_id', 'Diagnosis']].drop_duplicates()
merged_obs = pd.merge(pivot, diagnoses, on='care_calendar_id', how='left')

# Merge with Rx
full_data = pd.merge(merged_obs, rx_df, on='Diagnosis', how='inner')

full_data = full_data.rename(columns={
    'Diagnosis': 'diagnosis',
    'Breed': 'breed',
    'No. of Calvings': 'num_calvings',
    'Age as of 24-Feb-2025': 'age',
    'Months Pregnant as of 24-Feb-2025': 'months_pregnant',
    'Months Since Calving As of 24-Feb-2025': 'months_since_calving',
    'Average Animal LPD': 'avg_lpd'
})

required_cols = ['diagnosis', 'breed', 'num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd', 'Medicine']
full_data = full_data.dropna(subset=required_cols)

le = LabelEncoder()
full_data['Medicine_enc'] = le.fit_transform(full_data['Medicine'])

X = pd.get_dummies(full_data[['diagnosis', 'breed', 'num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd']])
y = full_data['Medicine_enc']

model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
model.fit(X, y)

feature_columns = X.columns.tolist()
unique_diagnoses = sorted(full_data['diagnosis'].unique())
unique_breeds = sorted(full_data['breed'].unique())

def predict_medicines():
    inputs = {
        "breed": breed_var.get(),
        "num_calvings": num_calvings_entry.get(),
        "age": age_entry.get(),
        "months_pregnant": months_pregnant_entry.get(),
        "months_since_calving": months_since_calving_entry.get(),
        "avg_lpd": avg_lpd_entry.get()
    }

    selected_indices = diagnosis_listbox.curselection()
    if not selected_indices:
        messagebox.showerror("Input Error", "Please select at least one diagnosis.")
        return

    diagnoses_list = [diagnosis_listbox.get(i) for i in selected_indices]

    rows = []
    for diag in diagnoses_list:
        row = inputs.copy()
        row["diagnosis"] = diag
        rows.append(row)

    user_df = pd.DataFrame(rows)
    numeric_fields = ['num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd']
    user_df[numeric_fields] = user_df[numeric_fields].apply(pd.to_numeric, errors='coerce')

    user_encoded = pd.get_dummies(user_df)
    for col in feature_columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0
    user_encoded = user_encoded[feature_columns]

    output_text.delete("1.0", END)

    probs = model.predict_proba(user_encoded)

    recommended_meds = set()
    output_text.insert(END, f"\n\n--- MEDICINES (By Confidence) ---\n\n")
    all_recommendations = []

    for i, diag in enumerate(diagnoses_list):
        sorted_indices = probs[i].argsort()[::-1]
        for idx in sorted_indices[:5]:  # You can change top 5 to top 3, top 10, etc.
            all_recommendations.append((i, idx))

    for i, idx in all_recommendations:
        med = le.inverse_transform([idx])[0]
        if med in recommended_meds:
            continue
        recommended_meds.add(med)
        prob = probs[i][idx]
        dose = full_data[full_data['Medicine'] == med]['Dosage'].mode()
        dose_str = dose.iloc[0] if not dose.empty else "Unknown"
        output_text.insert(END, f"Medicine: {med}\nDosage: {dose_str}\nConfidence: {prob * 100:.1f}%\n\n")

def save_model_files():
    try:
        joblib.dump(model, "cow_medicine_model.pkl")
        joblib.dump(le, "label_encoder.pkl")
        joblib.dump(feature_columns, "feature_columns.pkl")
        joblib.dump(unique_breeds, "unique_breeds.pkl")
        joblib.dump(unique_diagnoses, "unique_diagnoses.pkl")
        messagebox.showinfo("Success", "Model and encoders saved as .pkl files.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model: {e}")

# === Tkinter GUI ===
root = Tk()
root.title("Cow Medicine Predictor")

Label(root, text="Diagnosis (select multiple)").pack()
diagnosis_listbox = Listbox(root, selectmode=MULTIPLE, width=57, height=6)
diagnosis_listbox.pack()
diagnosis_scrollbar = Scrollbar(root, command=diagnosis_listbox.yview)
diagnosis_scrollbar.pack(side='right', fill='y')
diagnosis_listbox.config(yscrollcommand=diagnosis_scrollbar.set)

for diag in unique_diagnoses:
    diagnosis_listbox.insert('end', diag)

Label(root, text="Breed").pack()
breed_var = StringVar()
breed_combo = Combobox(root, textvariable=breed_var)
breed_combo['values'] = unique_breeds
breed_combo['width'] = 57
breed_combo.pack()

Label(root, text="Number of Calvings").pack()
num_calvings_entry = Entry(root, width=60)
num_calvings_entry.pack()

Label(root, text="Age").pack()
age_entry = Entry(root, width=60)
age_entry.pack()

Label(root, text="Months Pregnant").pack()
months_pregnant_entry = Entry(root, width=60)
months_pregnant_entry.pack()

Label(root, text="Months Since Calving").pack()
months_since_calving_entry = Entry(root, width=60)
months_since_calving_entry.pack()

Label(root, text="Average LPD").pack()
avg_lpd_entry = Entry(root, width=60)
avg_lpd_entry.pack()

Button(root, text="Predict Medicines", command=predict_medicines).pack(pady=10)
Button(root, text="Save Model (.pkl files)", command=save_model_files).pack(pady=5)

scrollbar = Scrollbar(root)
scrollbar.pack(side='right', fill='y')
output_text = Text(root, height=20, width=80, yscrollcommand=scrollbar.set)
output_text.pack()
scrollbar.config(command=output_text.yview)

root.mainloop()
