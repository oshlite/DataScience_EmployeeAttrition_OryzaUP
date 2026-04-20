from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import warnings
import os
import sys
import io
import requests
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')

print("[INFO] Flask app initialized")

# ====== LOAD MODEL & ARTIFACTS ======
def load_model_and_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        print("[DEBUG] Loading model from model/logistic_regression_best.pkl...")
        model = joblib.load('model/logistic_regression_best.pkl')
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None, None, None

    try:
        print("[DEBUG] Loading scaler and encoders...")
        scaler = joblib.load('model/scaler.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')
        print("[OK] Scaler and encoders loaded")
    except Exception as e:
        print(f"[ERROR] Scaler/encoders loading failed: {e}")
        scaler = None
        label_encoders = None

    return model, scaler, label_encoders

print("[DEBUG] Starting model initialization...")
model, scaler, label_encoders = load_model_and_artifacts()

# ====== INITIALIZE EXPERT SYSTEM ======
print("[DEBUG] Loading expert system...")
try:
    from expert_system import AttritionExpertSystem
    expert_system = AttritionExpertSystem()
    print("[OK] Expert system initialized")
except Exception as e:
    print(f"[ERROR] Expert system failed: {e}")
    expert_system = None

# ====== INITIALIZE DASHBOARD SERVICE ======
print("[DEBUG] Loading dashboard service...")
try:
    from dashboard_service import DashboardService
    dashboard_service = DashboardService('attrition_final.csv')
    print("[OK] Dashboard service initialized")
except Exception as e:
    print(f"[ERROR] Dashboard service failed: {e}")
    dashboard_service = None

# ====== LOAD DATA ======
def load_data():
    """Load employee data and predictions"""
    try:
        print("[DEBUG] Loading data...")
        df = pd.read_csv('attrition_final.csv')
        print(f"[OK] Loaded {len(df)} employees")
        
        try:
            predictions = pd.read_csv('attrition_predictions_output.csv')
            df = df.merge(predictions[['EmployeeId', 'Attrition_Probability', 'Prediction_Confidence', 'Predicted_Attrition']],
                          on='EmployeeId', how='left')
            print("[OK] Merged predictions")
        except:
            print("[WARN] Predictions file not found or merge failed")
        
        return df
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return None

print("[DEBUG] Loading data...")
df = load_data()

# ====== ROUTES ======

@app.route('/favicon.ico')
def favicon():
    """Return a simple favicon to avoid 404 errors"""
    return send_file(
        io.BytesIO(
            b'\x00\x00\x01\x00\x01\x00\x10\x10\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        ),
        mimetype='image/x-icon'
    )

@app.route('/')
def home():
    """Home page"""
    try:
        if df is not None:
            total_employees = len(df)
            attrition_count = int(df['Attrition'].sum())
            attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
        else:
            total_employees = 0
            attrition_count = 0
            attrition_rate = 0

        stats = {
            'total_employees': total_employees,
            'attrition_count': attrition_count,
            'attrition_rate': f"{attrition_rate:.1f}",
        }

        return render_template('index.html', stats=stats)
    except Exception as e:
        print(f"[ERROR] Home route failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error rendering home: {str(e)}", 500

@app.route('/test')
def test():
    """Simple test route"""
    return jsonify({
        'status': 'OK',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'expert_system_loaded': expert_system is not None,
        'dashboard_service_loaded': dashboard_service is not None,
        'data_loaded': df is not None
    })

@app.route('/prediction')
def prediction():
    """Prediction page"""
    try:
        return render_template('prediction.html')
    except Exception as e:
        print(f"[ERROR] Prediction route failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error rendering prediction: {str(e)}", 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint untuk Full ML Prediction dengan Explainability
    Probability: 100% dari Model ML (tanpa penyesuaian manual)
    Explanation: Expert system untuk interpretability & drivers
    """
    try:
        data = request.json
        print(f"\n[DEBUG] Received data: {data}")
        
        # Validate required fields
        required_fields = ['age', 'department', 'job_role', 'job_level', 'monthly_income', 
                          'overtime', 'job_satisfaction', 'work_life_balance', 
                          'years_at_company', 'years_in_role']
        
        missing_fields = [f for f in required_fields if not data.get(f)]
        if missing_fields:
            print(f"[ERROR] Missing required fields: {missing_fields}")
            return jsonify({'error': f'Required fields missing: {", ".join(missing_fields)}'}), 400
        
        print(f"[DEBUG] All required fields present")

        # Prepare input data with ALL features expected by model
        try:
            input_data = pd.DataFrame({
                'EmployeeId': [1],
                'Age': [int(data.get('age') or 35)],
                'BusinessTravel': [data.get('business_travel') or 'Travel_Rarely'],
                'DailyRate': [1000],
                'Department': [data.get('department') or 'Sales'],
                'DistanceFromHome': [int(data.get('distance_from_home') or 10)],
                'Education': [int(data.get('education') or 3)],
                'EducationField': [data.get('education_field') or 'Life Sciences'],
                'EmployeeCount': [1],
                'EnvironmentSatisfaction': [int(data.get('env_satisfaction') or 3)],
                'Gender': [data.get('gender') or 'Male'],
                'HourlyRate': [65],
                'JobInvolvement': [int(data.get('job_involvement') or 3)],
                'JobLevel': [int(data.get('job_level') or 2)],
                'JobRole': [data.get('job_role') or 'Sales Executive'],
                'JobSatisfaction': [int(data.get('job_satisfaction') or 3)],
                'MaritalStatus': [data.get('marital_status') or 'Married'],
                'MonthlyIncome': [int(data.get('monthly_income') or 5000)],
                'MonthlyRate': [2000],
                'NumCompaniesWorked': [int(data.get('num_companies_worked') or 2)],
                'Over18': ['Y'],
                'OverTime': [data.get('overtime') or 'No'],
                'PercentSalaryHike': [int(data.get('percent_salary_hike') or 12)],
                'PerformanceRating': [int(data.get('performance_rating') or 3)],
                'RelationshipSatisfaction': [int(data.get('relationship_satisfaction') or 3)],
                'StandardHours': [8],
                'StockOptionLevel': [int(data.get('stock_option_level') or 1)],
                'TotalWorkingYears': [int(data.get('total_working_years') or 10)],
                'TrainingTimesLastYear': [int(data.get('training_times') or 3)],
                'WorkLifeBalance': [int(data.get('work_life_balance') or 3)],
                'YearsAtCompany': [int(data.get('years_at_company') or 5)],
                'YearsInCurrentRole': [int(data.get('years_in_role') or 3)],
                'YearsSinceLastPromotion': [int(data.get('years_since_promotion') or 2)],
                'YearsWithCurrManager': [int(data.get('years_with_manager') or 2)],
                'Predicted_Attrition': [0],
                'Attrition_Final': [0]
            })
        except (ValueError, TypeError) as e:
            print(f"[ERROR] Data conversion error: {e}")
            print(f"[DEBUG] Received data: {data}")
            return jsonify({'error': f'Invalid data format: {str(e)}, check your input values'}), 400

        # Make prediction
        if model is None or scaler is None:
            return jsonify({'error': 'Model tidak dimuat'}), 500

        # Encode categorical columns using loaded label encoders
        categorical_cols = input_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except:
                    input_data[col] = 0

        # Ensure column order matches scaler expectations
        input_data = input_data[scaler.feature_names_in_]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Try to get probability, handle sklearn version compatibility
        try:
            model_probability = model.predict_proba(input_scaled)[0][1]
            print(f"[DEBUG] Using predict_proba, probability: {model_probability}")
        except AttributeError as e:
            print(f"[DEBUG] predict_proba failed ({str(e)}), using fallback method")
            # Fallback: use decision_function if available, or use prediction score
            try:
                # For logistic regression, we can use decision_function
                if hasattr(model, 'decision_function'):
                    decision = model.decision_function(input_scaled)[0]
                    # Convert decision score to probability-like value [0,1]
                    model_probability = 1 / (1 + np.exp(-decision))
                else:
                    # Simple fallback: 0.7 if prediction==1, 0.3 if prediction==0
                    model_probability = 0.75 if prediction == 1 else 0.25
                print(f"[DEBUG] Fallback probability: {model_probability}")
            except Exception as e2:
                print(f"[DEBUG] Fallback also failed: {e2}, using default")
                model_probability = 0.75 if prediction == 1 else 0.25

        # ===== FULL ML: Langsung pakai model probability =====
        final_probability = model_probability

        # Determine risk level based on ML probability
        if final_probability > 0.60:
            risk_level = "TINGGI"
            emoji = "🔴"
        elif final_probability > 0.30:
            risk_level = "SEDANG"
            emoji = "🟡"
        else:
            risk_level = "RENDAH"
            emoji = "🟢"

        # ===== EXPERT SYSTEM: Untuk explanation & drivers (bukan adjust probability) =====
        expert_data = {
            'age': int(data.get('age', 35)),
            'job_satisfaction': int(data.get('job_satisfaction', 3)),
            'overtime': data.get('overtime', 'No'),
            'years_since_promotion': int(data.get('years_since_promotion', 2)),
            'env_satisfaction': int(data.get('env_satisfaction', 3)),
            'work_life_balance': int(data.get('work_life_balance', 3)),
            'monthly_income': int(data.get('monthly_income', 5000)),
            'years_in_role': int(data.get('years_in_role', 3)),
            'relationship_satisfaction': int(data.get('relationship_satisfaction', 3)),
            'job_level': int(data.get('job_level', 2)),
            'dept_avg_income': 6000,
            'num_companies_worked': int(data.get('num_companies_worked', 2))
        }

        # Get expert analysis untuk explanation & drivers (used for interpretability only)
        diagnosis = expert_system.predict(final_probability, expert_data)

        # Override risk level dengan yang dari ML (bukan dari expert system)
        diagnosis['risk_level'] = risk_level
        diagnosis['emoji'] = emoji

        return jsonify({
            'success': True,
            'prediction': 'AT RISK' if prediction == 1 else 'STABLE',
            'ml_probability': f"{final_probability:.1%}",
            'ml_probability_value': float(final_probability),
            'risk_level': risk_level,
            'emoji': emoji,
            'top_drivers': diagnosis['top_drivers'],
            'recommendations': diagnosis['recommendations'],
            'contradiction_warning': diagnosis.get('contradiction_warning'),
            'additional_insights': diagnosis.get('additional_insights', [])
        })

    except Exception as e:
        print(f"\n[ERROR] Exception in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400

@app.route('/dashboard')
def dashboard():
    """Dashboard - Embed high-performance GitHub Pages version"""
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def get_dashboard_data():
    """API endpoint untuk dashboard charts"""
    if dashboard_service is None:
        return jsonify({'error': 'Dashboard service not available'}), 500

    try:
        charts = dashboard_service.get_all_charts()
        stats = dashboard_service.get_dashboard_stats()
        return jsonify({
            'success': True,
            'stats': stats,
            'charts': charts
        }), 200
    except Exception as e:
        print(f"Dashboard API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-dashboards-pdf')
def download_dashboards_pdf():
    """Download all Tableau dashboards as PDF"""
    try:
        print("[DEBUG] Starting PDF generation...")
        # Dashboard URLs
        dashboard_urls = [
            "https://public.tableau.com/app/profile/oryza.surya.hapsari/viz/dashboardya/DashboardHRTechTechCoreTech",
            "https://public.tableau.com/app/profile/oryza.surya.hapsari/viz/dashboardyaa/Dashboard",
            "https://public.tableau.com/app/profile/oryza.surya.hapsari/viz/dashboardya3/dashboardku",
            "https://public.tableau.com/app/profile/oryza.surya.hapsari/viz/dashboardya/ok",
            "https://public.tableau.com/app/profile/oryza.surya.hapsari/viz/dashboardya3/dashboardkuu",
            "https://public.tableau.com/app/profile/oryza.surya.hapsari/viz/dashboardya/end"
        ]
        
        # Create PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#000000',
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        title = Paragraph("Dashboard Attrition - HR Analytics", title_style)
        elements.append(title)
        
        date_style = ParagraphStyle(
            'CustomDate',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666666',
            spaceAfter=20,
            alignment=TA_CENTER
        )
        date_text = Paragraph(f"{datetime.now().strftime('%B %d, %Y')}", date_style)
        elements.append(date_text)
        elements.append(Spacer(1, 0.3*inch))
        
        # Try to get static images for each dashboard
        image_urls = [
            "https://public.tableau.com/static/images/da/dashboardya/DashboardHRTechTechCoreTech/1.png",
            "https://public.tableau.com/static/images/da/dashboardyaa/Dashboard/1.png",
            "https://public.tableau.com/static/images/da/dashboardya3/dashboardku/1.png",
            "https://public.tableau.com/static/images/da/dashboardya/ok/1.png",
            "https://public.tableau.com/static/images/da/dashboardya3/dashboardkuu/1.png",
            "https://public.tableau.com/static/images/da/dashboardya/end/1.png"
        ]
        
        dashboard_names = [
            "Dashboard HR Tech Tech CoreTech",
            "",
            "",
            "",
            "",
            ""
        ]
        
        for idx, (url, img_url, name) in enumerate(zip(dashboard_urls, image_urls, dashboard_names)):
            try:
                # Add dashboard title
                dash_title_style = ParagraphStyle(
                    'DashTitle',
                    parent=styles['Heading2'],
                    fontSize=14,
                    textColor='#ffb81c',
                    spaceAfter=10
                )
                elements.append(Paragraph(name, dash_title_style))
                
                # Try to fetch dashboard static image
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    img_data = io.BytesIO(response.content)
                    img = Image(img_data, width=6.5*inch, height=4.5*inch)
                    elements.append(img)
                else:
                    # Fallback text
                    elements.append(Paragraph(f"Dashboard: <a href='{url}' color='#ffb81c'>{name}</a>", styles['Normal']))
                    print(f"[WARN] Failed to load image for {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                # Fallback to link
                print(f"[WARN] Error loading dashboard {name}: {str(e)}")
                elements.append(Paragraph(f"Dashboard: <a href='{url}' color='#ffb81c'>{name}</a>", styles['Normal']))
            
            # Add page break between dashboards
            if idx < len(dashboard_urls) - 1:
                elements.append(PageBreak())
        
        # Build PDF
        print("[DEBUG] Building PDF document...")
        doc.build(elements)
        pdf_buffer.seek(0)
        
        print("[OK] PDF generated successfully")
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Dashboards_Attrition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        print(f"[ERROR] PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
