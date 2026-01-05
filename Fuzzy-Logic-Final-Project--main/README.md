# 🧠 Fuzzy Logic Health Risk Prediction System  
### Final Project – COMP 3106 (Introduction to Artificial Intelligence)

This project implements a **complete fuzzy logic–based cardiovascular risk prediction system** with:

- ✔ A full **Streamlit interactive dashboard**
- ✔ A backend **Mamdani fuzzy inference engine**
- ✔ Automatic dataset scoring & analysis
- ✔ Visual evaluation metrics, heatmaps, and histograms
- ✔ Explainable risk classification (Low / Medium / High)

The system allows **both single-patient prediction** and **population-level analysis**.

---

# 📁 Project Structure

```
Fuzzy-Logic-Health-Risk-Dashboard/
│
├── app.py                    # Streamlit dashboard (UI)
├── fuzzy_model.py            # Backend fuzzy engine
├── cleaned_risk_data.csv     # Raw dataset
├── cleaned_risk_with_fuzzy.csv   # Auto-generated scored dataset (created by code)
└── README.md                 # Documentation
```

---

# 🚀 Features

### 🔹 **Fuzzy Logic Engine (Backend)**
Defined in `fuzzy_model.py`:

- Fuzzy antecedents:
  - BMI  
  - Blood Pressure  
  - Abdominal Circumference  
  - Smoking  
  - Total Cholesterol  
- Consequent: **Cardiovascular Risk (0–100)**  
- Includes **over 30 fuzzy rules** (High / Medium / Low categories)
- Full Mamdani inference + centroid defuzzification
- Safe fallback to avoid simulation errors
- Classification into:
  - **Low (<40)**  
  - **Medium (40–70)**  
  - **High (70–100)**  

---

# 📊 Streamlit Dashboard (Frontend)

Defined in `app.py`.

The dashboard includes **three tabs**:

---

## 1️⃣ ⚡ Single Assessment (Patient-Level)
- Sidebar sliders for:
  - BMI  
  - Systolic BP  
  - Abdominal circumference  
  - Smoking status  
  - Total Cholesterol  
- Runs fuzzy inference  
- Displays:
  - 📈 Risk Score (0–100)  
  - 🔎 Risk Category  
  - ✔ Input summary snapshot

---

## 2️⃣ 📊 Dataset Analytics
- Loads raw dataset (`cleaned_risk_data.csv`)
- Shows:
  - Interactive histograms for any numeric variable  
  - Correlation heatmap  
  - Preview of first 25 rows  
- Auto-caching for faster performance  

---

## 3️⃣ 📈 Model Evaluation
- Loads the dataset scored with fuzzy logic  
- Provides:
  - Paginated table of:
    - BMI  
    - BP  
    - Abdominal circumference  
    - Cholesterol  
    - FuzzyRiskScore  
    - FuzzyRiskLevel  
  - Average risk based on:
    - Smoking  
    - Waist size groups  
  - Scatter plots:
    - Risk vs BMI  
    - Risk vs Blood Pressure  
  - Boxplot by Risk Level  
- If `CVD RISK` column exists:
  - Computes **MAE** and **R²** using sklearn (if installed)

---

# 🛠 How to Run the Project

### 1. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib scikit-fuzzy
```

Optional (for MAE/R² metrics):

```bash
pip install scikit-learn
```

---

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

This launches an interactive web dashboard in your browser.

---

### 3. Optional: Run backend only (console test)

```bash
python fuzzy_model.py
```

This prints the first few lines of the *scored dataset*.

---

# 📁 Dataset Information

`cleaned_risk_data.csv` must include the following columns:

- **BMI**
- **BLOOD PRESSURE**
- **ABDOMINAL CIRCUMFERENCE**
- **SMOKING**
- **TOTAL CHOLESTEROL**
- *(Optional)* **CVD RISK** — used only for sklearn evaluation

The system automatically generates:

`cleaned_risk_with_fuzzy.csv`  
with added columns:

- **FuzzyRiskScore**
- **FuzzyRiskLevel**

---

# 🔧 Customization

### ➤ Modify membership functions  
Inside:

```
fuzzy_model.py
```

### ➤ Modify fuzzy rules  
Search for:

```
# High risk rules
# Medium risk rules
# Low risk rules
```

### ➤ Change UI layout  
Customize:

```
app.py → Streamlit components
```

---

# 👨‍💻 Authors
**Christer Henrysson – Data Engineer & Model Evaluation Specialist &  AI Logic Architect 
Tirth Raval – AI Logic Architect
Samer Darraj – Frontend Engineer & UI/UX Systems Designer**

COMP 3106 – Carleton University (2025)

---

# 📄 License

This project is for **educational use** only.  
Feel free to extend or improve it.
