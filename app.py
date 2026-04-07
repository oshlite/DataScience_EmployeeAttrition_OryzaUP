import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
import joblib
from mlflow_config import setup_mlflow

# Setup MLflow
setup_mlflow()

# ====== LOAD MODEL & PREPROCESSING ARTIFACTS ======
@st.cache_resource
def load_model_and_artifacts():
    """Load trained model and preprocessing artifacts from MLflow"""
    try:
        # Try loading from MLflow registry
        model = mlflow.sklearn.load_model("models:/Employee_Attrition_Model/1")
    except:
        # Fallback to saved model file
        model = joblib.load('model/logistic_regression_best.pkl')

    scaler = joblib.load('model/scaler.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')

    return model, scaler, label_encoders

model, scaler, label_encoders = load_model_and_artifacts()

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM STYLING - LIGHT FUTURISTIC ======
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Main theme - Light futuristic */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #f0f4f9 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f7fb 100%);
        border-right: 1px solid #e8ecf1;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e8ecf1;
        border-left: 4px solid;
        color: #1a202c;
        box-shadow: 0 2px 8px rgba(100, 130, 180, 0.08);
    }

    /* Text colors */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Sidebar text */
    .stSidebar label {
        color: #1e293b !important;
        font-weight: 500 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Main text */
    body, p, div {
        color: #334155;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Section headers */
    .section-header {
        color: #0f172a !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
        font-family: 'Poppins', sans-serif !important;
    }

</style>
""", unsafe_allow_html=True)

# ====== LOAD DATA ======
@st.cache_data
def load_data():
    df = pd.read_csv('attrition_final.csv')
    predictions = pd.read_csv('attrition_predictions_output.csv')
    df = df.merge(predictions[['EmployeeId', 'Attrition_Probability', 'Prediction_Confidence', 'Predicted_Attrition']],
                  on='EmployeeId', how='left')
    return df

df = load_data()

# ====== SIDEBAR FILTERS ======
st.sidebar.markdown("<h3 style='color: #0f172a; margin-top: -10px;'>Filter Data</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

departments = ['All'] + sorted(df['Department'].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", departments, label_visibility="collapsed")

if selected_dept != 'All':
    job_roles = ['All'] + sorted(df[df['Department'] == selected_dept]['JobRole'].unique().tolist())
else:
    job_roles = ['All'] + sorted(df['JobRole'].unique().tolist())
selected_role = st.sidebar.selectbox("Job Role", job_roles, label_visibility="collapsed")

genders = ['All'] + sorted(df['Gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", genders, label_visibility="collapsed")

age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max), label_visibility="collapsed")

income_min, income_max = int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max())
income_range = st.sidebar.slider("Monthly Income Range", income_min, income_max, (income_min, income_max), label_visibility="collapsed")

st.sidebar.markdown("---")

# ====== APPLY FILTERS ======
filtered_df = df.copy()

if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]

if selected_role != 'All':
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
filtered_df = filtered_df[(filtered_df['MonthlyIncome'] >= income_range[0]) & (filtered_df['MonthlyIncome'] <= income_range[1])]
filtered_df = filtered_df[filtered_df['Attrition'].notna()]

# ====== MAIN TITLE ======
st.markdown("<h1 style='margin-bottom: 5px; font-size: 32px;'>Employee Attrition Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 14px; margin-bottom: 20px;'>Data-driven insights for HR decision making</p>", unsafe_allow_html=True)

# ====== KPI CARDS ======
st.markdown("---")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

total_employees = len(filtered_df)
attrition_count = int(filtered_df['Attrition'].sum())
attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
avg_income = filtered_df['MonthlyIncome'].mean()

# ====== LOG METRICS TO MLFLOW ======
try:
    with mlflow.start_run(nested=True):
        mlflow.log_metric("total_employees", total_employees)
        mlflow.log_metric("attrition_count", attrition_count)
        mlflow.log_metric("attrition_rate", attrition_rate)
        mlflow.log_metric("avg_monthly_income", avg_income)
        # Log filter parameters
        mlflow.log_param("selected_department", selected_dept)
        mlflow.log_param("selected_role", selected_role)
        mlflow.log_param("selected_gender", selected_gender)
except Exception as e:
    st.warning(f"MLflow logging issue: {e}")

with kpi_col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #3b82f6;">
        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Total Employees</div>
        <div style="font-size: 40px; font-weight: 800; color: #0f172a; margin-top: 12px;">{total_employees:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    color = "#ef4444" if attrition_rate > 20 else "#10b981"
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Attrition Rate</div>
        <div style="font-size: 40px; font-weight: 800; color: {color}; margin-top: 12px;">{attrition_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #f97316;">
        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Employees Left</div>
        <div style="font-size: 40px; font-weight: 800; color: #f97316; margin-top: 12px;">{attrition_count}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #8b5cf6;">
        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Average Income</div>
        <div style="font-size: 36px; font-weight: 800; color: #8b5cf6; margin-top: 12px;">${avg_income/1000:.1f}K</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ====== ROW 1: DEPARTMENT & ROLE ======
st.markdown("<h3 class='section-header'>Attrition by Department and Role</h3>", unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    dept_attrition = filtered_df.groupby('Department')['Attrition'].agg(['sum', 'count'])
    dept_attrition['rate'] = (dept_attrition['sum'] / dept_attrition['count'] * 100).round(1)
    dept_attrition = dept_attrition.sort_values('rate', ascending=True).reset_index()

    fig1 = go.Figure(data=[
        go.Bar(x=dept_attrition['rate'],
               y=dept_attrition['Department'],
               orientation='h',
               text=dept_attrition['rate'].apply(lambda x: f'{x:.1f}%'),
               textposition='outside',
               marker=dict(
                   color=dept_attrition['rate'],
                   colorscale='RdYlGn_r',
                   showscale=False,
                   line=dict(color='white', width=0)
               ))
    ])
    fig1.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=11, family='Poppins'),
        height=350,
        margin=dict(l=120, r=50, t=20, b=20),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        hovermode='closest'
    )
    fig1.update_yaxes(autorange="reversed")
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    role_attrition = filtered_df.groupby('JobRole')['Attrition'].agg(['sum', 'count'])
    role_attrition['rate'] = (role_attrition['sum'] / role_attrition['count'] * 100).round(1)
    role_attrition = role_attrition.sort_values('rate', ascending=True).tail(10).reset_index()

    fig2 = go.Figure(data=[
        go.Bar(x=role_attrition['rate'],
               y=role_attrition['JobRole'],
               orientation='h',
               text=role_attrition['rate'].apply(lambda x: f'{x:.1f}%'),
               textposition='outside',
               marker=dict(
                   color=role_attrition['rate'],
                   colorscale='RdYlGn_r',
                   showscale=False,
                   line=dict(color='white', width=0)
               ))
    ])
    fig2.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=11, family='Poppins'),
        height=350,
        margin=dict(l=150, r=50, t=20, b=20),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        hovermode='closest'
    )
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ====== ROW 2: AGE GROUP & TENURE ======
st.markdown("<h3 class='section-header'>Attrition by Demographics</h3>", unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'],
                                     bins=[0, 25, 35, 45, 55, 100],
                                     labels=['18-25', '25-35', '35-45', '45-55', '55+'])
    age_attrition = filtered_df.groupby('AgeGroup')['Attrition'].agg(['sum', 'count'])
    age_attrition['rate'] = (age_attrition['sum'] / age_attrition['count'] * 100).round(1)
    age_attrition = age_attrition.reset_index()

    fig3 = px.bar(age_attrition, x='AgeGroup', y='rate', text='rate',
                  color='rate', color_continuous_scale='Blues',
                  title='')
    fig3.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig3.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=11, family='Poppins'),
        height=350,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False,
        hovermode='closest'
    )
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    filtered_df['TenureGroup'] = pd.cut(filtered_df['YearsAtCompany'],
                                        bins=[0, 1, 3, 5, 10, 100],
                                        labels=['0-1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'])
    tenure_attrition = filtered_df.groupby('TenureGroup')['Attrition'].agg(['sum', 'count'])
    tenure_attrition['rate'] = (tenure_attrition['sum'] / tenure_attrition['count'] * 100).round(1)
    tenure_attrition = tenure_attrition.reset_index()

    fig4 = px.bar(tenure_attrition, x='TenureGroup', y='rate', text='rate',
                  color='rate', color_continuous_scale='Reds',
                  title='')
    fig4.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig4.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=11, family='Poppins'),
        height=350,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False,
        hovermode='closest'
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ====== ROW 3: INCOME & TRAVEL ======
st.markdown("<h3 class='section-header'>Attrition by Income and Travel</h3>", unsafe_allow_html=True)

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    filtered_df['IncomeGroup'] = pd.cut(filtered_df['MonthlyIncome'],
                                        bins=[0, 3000, 6000, 10000, 20000],
                                        labels=['<3K', '3-6K', '6-10K', '10K+'])
    income_attrition = filtered_df.groupby('IncomeGroup')['Attrition'].agg(['sum', 'count'])
    income_attrition['rate'] = (income_attrition['sum'] / income_attrition['count'] * 100).round(1)
    income_attrition = income_attrition.reset_index()

    fig5 = px.bar(income_attrition, x='IncomeGroup', y='rate', text='rate',
                  color='rate', color_continuous_scale='Purples',
                  title='')
    fig5.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig5.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=11, family='Poppins'),
        height=350,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False,
        hovermode='closest'
    )
    st.plotly_chart(fig5, use_container_width=True)

with row3_col2:
    travel_attrition = filtered_df.groupby('BusinessTravel')['Attrition'].agg(['sum', 'count'])
    travel_attrition['rate'] = (travel_attrition['sum'] / travel_attrition['count'] * 100).round(1)
    travel_attrition = travel_attrition.sort_values('rate', ascending=True).reset_index()

    fig6 = px.bar(travel_attrition, x='BusinessTravel', y='rate', text='rate',
                  color='rate', color_continuous_scale='Oranges',
                  title='')
    fig6.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig6.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=11, family='Poppins'),
        height=350,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False,
        hovermode='closest'
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ====== ROW 4: GENDER & MARITAL ======
st.markdown("<h3 class='section-header'>Attrition by Personal Factors</h3>", unsafe_allow_html=True)

row4_col1, row4_col2, row4_col3 = st.columns(3)

with row4_col1:
    gender_attrition = filtered_df.groupby('Gender')['Attrition'].agg(['sum', 'count'])
    gender_attrition['rate'] = (gender_attrition['sum'] / gender_attrition['count'] * 100).round(1)
    gender_attrition = gender_attrition.reset_index()

    fig_gender = px.bar(gender_attrition, x='Gender', y='rate', text='rate',
                        color='rate', color_continuous_scale='Greens',
                        title='')
    fig_gender.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig_gender.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=10, family='Poppins'),
        height=300,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with row4_col2:
    marital_attrition = filtered_df.groupby('MaritalStatus')['Attrition'].agg(['sum', 'count'])
    marital_attrition['rate'] = (marital_attrition['sum'] / marital_attrition['count'] * 100).round(1)
    marital_attrition = marital_attrition.reset_index()

    fig_marital = px.bar(marital_attrition, x='MaritalStatus', y='rate', text='rate',
                         color='rate', color_continuous_scale='teal',
                         title='')
    fig_marital.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig_marital.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=10, family='Poppins'),
        height=300,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False
    )
    st.plotly_chart(fig_marital, use_container_width=True)

with row4_col3:
    overtime_attrition = filtered_df.groupby('OverTime')['Attrition'].agg(['sum', 'count'])
    overtime_attrition['rate'] = (overtime_attrition['sum'] / overtime_attrition['count'] * 100).round(1)
    overtime_attrition = overtime_attrition.reset_index()

    fig_ot = px.bar(overtime_attrition, x='OverTime', y='rate', text='rate',
                    color='rate', color_continuous_scale='Reds',
                    title='')
    fig_ot.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig_ot.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=10, family='Poppins'),
        height=300,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False
    )
    st.plotly_chart(fig_ot, use_container_width=True)

st.markdown("---")

# ====== ROW 5: PREDICTION ANALYTICS ======
st.markdown("<h3 class='section-header'>Prediction & Risk Analysis</h3>", unsafe_allow_html=True)

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    fig_hist = px.histogram(filtered_df, x='Attrition_Probability', nbins=30,
                            color_discrete_sequence=['#3b82f6'], title='')
    fig_hist.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=10, family='Poppins'),
        height=300,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title='Risk Score'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title='Count'),
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with pred_col2:
    filtered_df['RiskCategory'] = pd.cut(filtered_df['Attrition_Probability'],
                                         bins=[0, 0.33, 0.66, 1],
                                         labels=['Low', 'Medium', 'High'])
    risk_counts = filtered_df['RiskCategory'].value_counts()

    fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index, title='',
                      color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'})
    fig_risk.update_layout(
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=10, family='Poppins'),
        height=300,
        margin=dict(l=0, r=0, t=20, b=20)
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with pred_col3:
    actual_dist = filtered_df['Attrition'].value_counts().reset_index()
    actual_dist.columns = ['Status', 'Count']
    actual_dist['Status'] = actual_dist['Status'].map({0.0: 'Stayed', 1.0: 'Left'})

    fig_actual = px.bar(actual_dist, x='Status', y='Count', text='Count',
                        color='Status', color_discrete_map={'Stayed': '#10b981', 'Left': '#ef4444'},
                        title='')
    fig_actual.update_traces(textposition='outside', texttemplate='%{text}')
    fig_actual.update_layout(
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(248,249,252,1)',
        font=dict(color='#334155', size=10, family='Poppins'),
        height=300,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, title=''),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False, title=''),
        showlegend=False
    )
    st.plotly_chart(fig_actual, use_container_width=True)

st.markdown("---")

# ====== EMPLOYEE ATTRITION PREDICTION ======
st.markdown("<h3 class='section-header'>Employee Attrition Prediction</h3>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 13px; margin-bottom: 20px;'>Predict attrition risk for a new employee or specific case</p>", unsafe_allow_html=True)

# Create input form
col1, col2 = st.columns(2)

with col1:
    years_at_company = st.slider("Years at Company", 0, 40, 5, help="Employee tenure")
    monthly_income = st.number_input("Monthly Income ($)", value=5000, min_value=1000, max_value=20000, step=500)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2, help="1=Low, 4=Very High")

with col2:
    age = st.slider("Age", 18, 65, 35, help="Employee age")
    distance_from_home = st.slider("Distance from Home (km)", 1, 30, 10)
    overtime = st.selectbox("Overtime", ["No", "Yes"], help="Works overtime?")

# Add prediction button
if st.button("Predict Attrition Risk", use_container_width=True, type="primary"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'YearsAtCompany': [years_at_company],
            'MonthlyIncome': [monthly_income],
            'JobSatisfaction': [job_satisfaction],
            'Age': [age],
            'DistanceFromHome': [distance_from_home],
            'OverTime': [1 if overtime == "Yes" else 0]
        })

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Display results
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color: {'#ef4444' if prediction == 1 else '#10b981'};'>
                <div style='font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;'>Prediction</div>
                <div style='font-size: 40px; font-weight: 800; color: #0f172a; margin-top: 12px;'>
                    {'AT RISK' if prediction == 1 else 'STABLE'}
                </div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 8px;'>Employee status</div>
            </div>
            """, unsafe_allow_html=True)

        with result_col2:
            st.markdown(f"""
            <div class='metric-card' style='border-left-color: {'#ef4444' if probability > 0.6 else '#f59e0b' if probability > 0.3 else '#10b981'};'>
                <div style='font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;'>Risk Probability</div>
                <div style='font-size: 40px; font-weight: 800; color: #0f172a; margin-top: 12px;'>{probability:.1%}</div>
                <div style='font-size: 12px; color: #94a3b8; margin-top: 8px;'>Likelihood of attrition</div>
            </div>
            """, unsafe_allow_html=True)

        # Risk interpretation
        if probability > 0.6:
            risk_level = "HIGH"
            color = "#ef4444"
            recommendation = "Immediate intervention recommended. Review compensation, role fit, and career development opportunities."
        elif probability > 0.3:
            risk_level = "MEDIUM"
            color = "#f59e0b"
            recommendation = "Monitor closely. Consider engagement initiatives and career path discussions."
        else:
            risk_level = "LOW"
            color = "#10b981"
            recommendation = "Employee appears satisfied. Continue regular engagement practices."

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(248,249,252,0.8), rgba(240,244,249,0.8));
                    padding: 20px; border-left: 4px solid {color}; border-radius: 8px; margin-top: 20px;'>
            <p><strong>Risk Level:</strong> <span style='color: {color};'>{risk_level}</span></p>
            <p><strong>Recommendation:</strong> {recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

        # Log prediction to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_param("years_at_company", years_at_company)
            mlflow.log_param("monthly_income", monthly_income)
            mlflow.log_param("job_satisfaction", job_satisfaction)
            mlflow.log_param("age", age)
            mlflow.log_param("distance_from_home", distance_from_home)
            mlflow.log_param("overtime", overtime)
            mlflow.log_metric("prediction", prediction)
            mlflow.log_metric("probability", probability)

    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.markdown("---")

st.markdown("<h3 class='section-header'>High Risk Employees</h3>", unsafe_allow_html=True)

high_risk = filtered_df.nlargest(30, 'Attrition_Probability')[
    ['EmployeeId', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany',
     'Attrition_Probability', 'Attrition', 'JobSatisfaction', 'EnvironmentSatisfaction']
].copy()

high_risk['Status'] = high_risk['Attrition'].map({0: 'Stayed', 1: 'Left', np.nan: 'Unknown'})
high_risk = high_risk.drop('Attrition', axis=1)
high_risk = high_risk.rename(columns={
    'EmployeeId': 'ID',
    'Department': 'Department',
    'JobRole': 'Role',
    'MonthlyIncome': 'Income',
    'YearsAtCompany': 'Tenure',
    'Attrition_Probability': 'Risk Score',
    'JobSatisfaction': 'Job Sat',
    'EnvironmentSatisfaction': 'Env Sat'
})

high_risk['Risk Score'] = high_risk['Risk Score'].apply(lambda x: f"{x:.1%}")
high_risk['Income'] = high_risk['Income'].apply(lambda x: f"${x:,.0f}")

st.dataframe(
    high_risk,
    use_container_width=True,
    hide_index=True,
    column_config={
        'ID': st.column_config.NumberColumn(width=60),
        'Department': st.column_config.TextColumn(width=100),
        'Role': st.column_config.TextColumn(width=120),
        'Income': st.column_config.TextColumn(width=90),
        'Tenure': st.column_config.NumberColumn(width=70),
        'Risk Score': st.column_config.TextColumn(width=90),
    }
)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>Dashboard updated with real-time data | Data source: HR system</p>", unsafe_allow_html=True)
