# ❤️ Heart Disease Prediction Using Machine Learning

> A complete industry-level machine learning project for engineering students.

---

## 📁 Project Structure

```
Heart_Disease_Prediction/
├── dataset/
│   └── heart.csv               ← Place your dataset here (auto-generated if missing)
├── models/
│   ├── best_model.pkl          ← Saved trained model (after running train_model.py)
│   ├── scaler.pkl              ← Feature scaler
│   ├── feature_names.pkl       ← Feature column names
│   └── model_metadata.json     ← Model metrics & config
├── screenshots/
│   ├── 01_disease_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_cholesterol_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_feature_importance.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   └── 08_model_comparison.png
├── notebooks/
│   └── (optional Jupyter notebooks for EDA)
├── app.py                      ← Streamlit web application
├── train_model.py              ← ML training pipeline
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## ⚙️ Setup & Run (Windows — VS Code)

### Step 1 — Install Python
Download Python 3.10+ from https://python.org and check "Add to PATH" during install.

### Step 2 — Open in VS Code
```
File → Open Folder → Heart_Disease_Prediction
```

### Step 3 — Create Virtual Environment
```bash
# In VS Code Terminal (Ctrl + `)
python -m venv venv
venv\Scripts\activate
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Train the Model
```bash
python train_model.py
```
This will:
- Load / generate the dataset
- Perform EDA and preprocessing
- Train 5 ML models
- Save all 8 visualization plots to `screenshots/`
- Save the best model to `models/`

### Step 6 — Run the Web App
```bash
streamlit run app.py
```
Open your browser at → http://localhost:8501

---

## 📊 Dataset

The project uses the **Cleveland Heart Disease Dataset** (UCI Repository).
- **Rows:** 1,025 patients
- **Columns:** 13 features + 1 target
- **Target:** `1` = Heart Disease present, `0` = No disease

---

## 🤖 Models Trained

| Model               | Notes                        |
|---------------------|------------------------------|
| Logistic Regression | Baseline linear classifier   |
| Decision Tree       | Interpretable, depth=5       |
| Random Forest       | Ensemble, 100 trees          |
| KNN                 | Instance-based, k=7          |
| SVM (RBF)           | Kernel-based, high accuracy  |

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- Cross-Validation Score (5-fold Stratified)
- AUC-ROC
- Confusion Matrix
- Classification Report

---

## 🌐 Deployment

### GitHub
```bash
git init
git add .
git commit -m "Initial commit — Heart Disease Prediction"
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git
git push -u origin main
```

### Streamlit Cloud
1. Push code to GitHub (include `requirements.txt`)
2. Go to https://streamlit.io/cloud
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> ⚠️ Before deploying: run `train_model.py` locally first and include the `models/` folder in your repo (or add a `setup.py` that auto-trains on first run).

---

## 📝 License
For educational use only. Not for clinical diagnosis.
