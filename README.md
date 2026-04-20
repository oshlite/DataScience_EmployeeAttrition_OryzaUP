# 👔 Employee Attrition Prediction System

**Sistem prediksi risiko attrition karyawan berbasis Machine Learning + Expert System**

---

## 📊 Fitur Utama

- ✅ **Model ML dengan Akurasi 85%** - Logistic Regression dilatih pada 1.470 data karyawan
- ✅ **10 Field Penting** - Assessment form yang kompleks namun fokus pada driver utama attrition
- ✅ **Expert System** - Identifikasi faktor risiko spesifik (kepuasan kerja, overtime, gaji, promosi)
- ✅ **Rekomendasi Aksi HR** - Actionable insights berdasarkan profile karyawan
- ✅ **Dashboard Interaktif** - Tableau dashboards dengan visualisasi attrition pattern
- ✅ **Bilingual** - Bahasa Indonesia + English

---

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone repository
git clone https://github.com/your-username/employee-attrition.git
cd employee-attrition

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env dengan credentials DagsHub (optional)

# 5. Run Flask app
python web_app.py
# Open http://localhost:5000
```

---

## 📋 Dataset

**attrition_final.csv** - 1.470 karyawan dengan 30+ features:
- Demographic: Age, Gender, Marital Status
- Job-related: Department, Role, Level, Income, Satisfaction
- Work factors: Overtime, Travel, Work-Life Balance
- Career: Promotion, Manager, Training
- Target: Attrition (Yes/No)

---

## 🏗️ Architecture

```
project/
├── web_app.py              → Flask application (main entry)
├── app.py                  → Streamlit app (alternative)
├── model/
│   ├── train.py           → Model training script
│   ├── logistic_regression_best.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── expert_system.py        → HR expert rules engine
├── dashboard_service.py    → Dashboard analytics
├── templates/              → HTML templates
├── static/                 → CSS, JS, assets
└── attrition_final.csv     → Training data
```

---

## 🎯 Assessment Form (10 Core Fields)

**Required:**
1. **Age** - Employee age (18-80 years)
2. **Department** - Sales / R&D / HR
3. **Job Role** - Position title
4. **Job Level** - 1(Entry) to 5(Executive)
5. **Monthly Income** - Salary in USD
6. **Overtime** - Yes / No
7. **Job Satisfaction** - 1(Low) to 4(High)
8. **Work-Life Balance** - 1(Low) to 4(High)
9. **Years at Company** - Total tenure
10. **Years in Current Role** - Role tenure

**Optional Advanced:**
- Environment Satisfaction, Relationship, Performance
- Salary Hike, Stock Options, Training
- Previous Companies, Travel, Distance, etc.

---

## 🔧 Model Specifics

**Algorithm:** Logistic Regression (Binary Classification)
- **Training Data:** 1.470 examples, 30 features
- **Test Accuracy:** ~85%
- **Cross-validation:** 5-fold stratified
- **Class Imbalance Handling:** SMOTE resampling
- **Feature Scaling:** StandardScaler

**Risk Thresholds:**
- 0-30%: RENDAH (Low)
- 30-60%: SEDANG (Medium)
- 60%+: TINGGI (High)

---

## 📡 API Endpoints

### Flask Web App (Primary)
```
GET  /                    - Home page
GET  /prediction          - Assessment form
POST /api/predict         - ML prediction endpoint
GET  /dashboard           - Tableau dashboards
JSON response includes:
  - ml_probability (0-100%)
  - risk_level (RENDAH/SEDANG/TINGGI)
  - top_drivers (key risk factors)
  - recommendations (HR actions)
```

---

## 🌐 Deployment

### Railway Deployment (Recommended)

**Prerequisites:**
- GitHub account with repo pushed
- Railway account (free tier available)

**Steps:**

1. **Push to GitHub** (already done ✅)
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway:**
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub"
   - Select this repository
   - Railway auto-detects Procfile
   - Set environment variables:
     ```
     FLASK_ENV=production
     DAGSHUB_USER_NAME=your_username
     DAGSHUB_USER_TOKEN=your_token
     MLFLOW_TRACKING_URI=https://dagshub.com/your_username/employee-attrition/mlflow
     ```
   - Click "Deploy"

3. **Access deployed app:**
   - Railway generates a public URL
   - Example: `https://app-xyz.railway.app`

**Cost:** Free tier provides 5GB storage + $5/month credits

---

## 🔐 Environment Variables

Create `.env` file:
```
FLASK_ENV=production
DAGSHUB_USER_NAME=your_dagshub_username
DAGSHUB_USER_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/.../mlflow
```

See `.env.example` for template.

---

## 📚 Data Dictionary

Key fields explanation:

| Field | Description | Range |
|-------|-------------|-------|
| Age | Employee age | 18-80 |
| JobLevel | Organizational level | 1-5 |
| MonthlyIncome | Salary (USD) | 1000-20000 |
| JobSatisfaction | Satisfaction rating | 1-4 |
| OverTime | Works overtime | Yes/No |
| YearsAtCompany | Tenure (years) | 0-40 |
| YearsInCurrentRole | Current role tenure | 0-40 |

---

## 🛠️ Technology Stack

- **Backend:** Flask 3.1
- **ML:** scikit-learn 1.3.2, pandas, numpy
- **Frontend:** HTML/CSS/JS, Plotly dashboards
- **Database:** CSV (local), MLflow (optional)
- **Deployment:** Gunicorn, Railway
- **Optional:** Streamlit, DagsHub, MLflow

---

## 📝 Notes

- Model trained on balanced dataset using SMOTE
- Feature scaling applied (StandardScaler)
- Predictions cached to reduce computation
- All categorical variables label-encoded
- Salary/income data is fictional for demo

---

## 📞 Support

For issues:
1. Check browser console (F12) for errors
2. Review server logs on Railway dashboard
3. Verify environment variables are set
4. Ensure data files (CSV, PKL) are uploaded

---

## 📄 License

This project is confidential - for HR analytics demonstration only.

---

**Last Updated:** April 15, 2026  
**Version:** 1.0.0