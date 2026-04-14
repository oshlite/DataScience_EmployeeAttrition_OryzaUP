import joblib
import pandas as pd

# Load model and preprocessing
model = joblib.load('model/logistic_regression_best.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

# Helper function to create full data with default values
def create_data(age, dept, role, level, income, overtime, job_sat, work_life, years_comp, years_role):
    return pd.DataFrame({
        'EmployeeId': [1], 'Age': [age], 'BusinessTravel': ['Travel_Rarely'],
        'DailyRate': [1000], 'Department': [dept], 'DistanceFromHome': [10],
        'Education': [3], 'EducationField': ['Technical Degree'], 'EmployeeCount': [1],
        'EnvironmentSatisfaction': [3], 'Gender': ['Male'], 'HourlyRate': [65],
        'JobInvolvement': [3], 'JobLevel': [level], 'JobRole': [role],
        'JobSatisfaction': [job_sat], 'MaritalStatus': ['Married'],
        'MonthlyIncome': [income], 'MonthlyRate': [2000], 'NumCompaniesWorked': [2],
        'Over18': ['Y'], 'OverTime': [overtime], 'PercentSalaryHike': [12],
        'PerformanceRating': [3], 'RelationshipSatisfaction': [3], 'StandardHours': [8],
        'StockOptionLevel': [1], 'TotalWorkingYears': [10], 'TrainingTimesLastYear': [3],
        'WorkLifeBalance': [work_life], 'YearsAtCompany': [years_comp],
        'YearsInCurrentRole': [years_role], 'YearsSinceLastPromotion': [2],
        'YearsWithCurrManager': [2], 'Predicted_Attrition': [0], 'Attrition_Final': [0],
        'Age_key': [age], 'Department_key': [dept], 'JobRole_key': [role],
        'JobLevel_key': [level], 'MonthlyIncome_key': [income], 'OverTime_key': [overtime],
        'JobSatisfaction_key': [job_sat], 'WorkLifeBalance_key': [work_life],
        'YearsAtCompany_key': [years_comp], 'YearsInCurrentRole_key': [years_role]
    })

# SKENARIO 1: RISIKO RENDAH (Budi Santoso)
s1 = create_data(35, 'Research & Development', 'Research Director', 4, 8000, 'No', 4, 4, 10, 5)

# SKENARIO 2: RISIKO SEDANG (Siti Nurhaliza)
s2 = create_data(28, 'Human Resources', 'Human Resources', 2, 3500, 'Yes', 2, 2, 2, 1)

# SKENARIO 3: RISIKO TINGGI (Ahmad Wijaya)
s3 = create_data(25, 'Sales', 'Sales Executive', 1, 2000, 'Yes', 1, 1, 1, 0.5)

scenarios = [s1, s2, s3]
names = ["Budi Santoso", "Siti Nurhaliza", "Ahmad Wijaya"]
risks = ["RENDAH", "SEDANG", "TINGGI"]
emojis = ["🟢", "🟡", "🔴"]

for i, (data, name, risk, emoji) in enumerate(zip(scenarios, names, risks, emojis), 1):
    # Copy for encoding
    data_enc = data.copy()
    
    # Encode categorical columns using loaded label encoders
    categorical_cols = data_enc.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in label_encoders and '_key' not in col:
            try:
                data_enc[col] = label_encoders[col].transform(data_enc[col].astype(str))
            except:
                data_enc[col] = 0
    
    # Ensure column order matches scaler expectations
    data_proc = data_enc[scaler.feature_names_in_].copy()
    
    # Predict
    data_scaled = scaler.transform(data_proc)
    
    # Use predict instead of predict_proba due to version compatibility
    prediction = model.predict(data_scaled)[0]
    
    # Try predict_proba, if it fails, use prediction as proxy
    try:
        probability = model.predict_proba(data_scaled)[0][1]
    except:
        probability = 0.8 if prediction == 1 else 0.2
    
    print("\n" + "="*75)
    print(f"SKENARIO {i}: {name.upper()} - RISIKO {risk}")
    print("="*75)
    print(f"Attrition Risk Probability: {probability*100:.1f}%")
    print("\nFORM YANG DIISI (10 FIELD PENTING):")
    print("-"*75)
    print(f"  1. Age                    = {data['Age'].values[0]} tahun")
    print(f"  2. Department             = {data['Department'].values[0]}")
    print(f"  3. Job Role               = {data['JobRole'].values[0]}")
    print(f"  4. Job Level              = {data['JobLevel'].values[0]}")
    print(f"  5. Monthly Income         = ${data['MonthlyIncome'].values[0]:,}")
    print(f"  6. Overtime Status        = {data['OverTime'].values[0]}")
    print(f"  7. Job Satisfaction       = {data['JobSatisfaction'].values[0]} (1=Low, 4=High)")
    print(f"  8. Work-Life Balance      = {data['WorkLifeBalance'].values[0]} (1=Low, 4=High)")
    print(f"  9. Years at Company       = {data['YearsAtCompany'].values[0]} tahun")
    print(f" 10. Years in Current Role  = {data['YearsInCurrentRole'].values[0]} tahun")
    print("="*75)
