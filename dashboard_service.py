import pandas as pd
import json

class DashboardService:
    def __init__(self, csv_path="attrition_final.csv"):
        self.df = pd.read_csv(csv_path)
        self.df['RiskLevel'] = self._categorize_risk()

    def _categorize_risk(self):
        """Categorize attrition probability into risk levels"""
        def categorize(val):
            if val < 0.3:
                return 'Rendah'
            elif val < 0.6:
                return 'Sedang'
            else:
                return 'Tinggi'
        return self.df['Attrition_Final'].apply(categorize)

    def get_risk_distribution(self):
        """Risk level distribution data for chart"""
        risk_counts = self.df['RiskLevel'].value_counts()
        return {
            'x': risk_counts.index.tolist(),
            'y': risk_counts.values.tolist(),
            'type': 'bar',
            'marker': {
                'color': [{'Rendah': '#10b981', 'Sedang': '#f59e0b', 'Tinggi': '#ef4444'}.get(x, '#6b7280') for x in risk_counts.index]
            }
        }

    def get_risk_by_department(self):
        """Risk by department data"""
        dept_risk = self.df.groupby('Department')['Attrition_Final'].mean().sort_values(ascending=False)
        return {
            'x': dept_risk.index.tolist(),
            'y': dept_risk.values.tolist(),
            'type': 'bar',
            'marker': {'color': '#2563eb'}
        }

    def get_risk_by_age(self):
        """Risk by age groups"""
        self.df['AgeGroup'] = pd.cut(self.df['Age'],
                                     bins=[0, 25, 35, 45, 55, 100],
                                     labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        age_risk = self.df.groupby('AgeGroup')['Attrition_Final'].mean()

        return {
            'x': age_risk.index.astype(str).tolist(),
            'y': age_risk.values.tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'line': {'color': '#2563eb', 'width': 3},
            'marker': {'size': 10, 'color': '#2563eb'}
        }

    def get_satisfaction_heatmap(self):
        """Satisfaction vs Risk heatmap data"""
        satisfaction_risk = self.df.groupby(['JobSatisfaction', 'EnvironmentSatisfaction'])['Attrition_Final'].mean().reset_index()
        pivot = satisfaction_risk.pivot(index='EnvironmentSatisfaction',
                                       columns='JobSatisfaction',
                                       values='Attrition_Final')

        return {
            'z': pivot.values.tolist(),
            'x': [f'Tingkat {i}' for i in pivot.columns.tolist()],
            'y': [f'Tingkat {i}' for i in pivot.index.tolist()],
            'colorscale': 'RdYlGn_r',
            'type': 'heatmap'
        }

    def get_tenure_scatter(self):
        """Tenure vs Risk scatter plot"""
        colors_map = {'Rendah': '#10b981', 'Sedang': '#f59e0b', 'Tinggi': '#ef4444'}

        # Group by risk level for traces
        traces = []
        for risk_level in ['Rendah', 'Sedang', 'Tinggi']:
            mask = self.df['RiskLevel'] == risk_level
            subset = self.df[mask]
            traces.append({
                'x': subset['YearsAtCompany'].tolist(),
                'y': subset['Attrition_Final'].tolist(),
                'mode': 'markers',
                'type': 'scatter',
                'name': risk_level,
                'marker': {
                    'size': (subset['MonthlyIncome'] / 5000 * 8).tolist(),
                    'color': colors_map[risk_level],
                    'opacity': 0.7
                }
            })

        return traces

    def get_dashboard_stats(self):
        """Get summary statistics"""
        total = len(self.df)
        high_risk_count = (self.df['RiskLevel'] == 'Tinggi').sum()
        medium_risk_count = (self.df['RiskLevel'] == 'Sedang').sum()
        low_risk_count = (self.df['RiskLevel'] == 'Rendah').sum()
        avg_risk = self.df['Attrition_Final'].mean()

        return {
            'total_employees': int(total),
            'high_risk': int(high_risk_count),
            'medium_risk': int(medium_risk_count),
            'low_risk': int(low_risk_count),
            'avg_risk': round(avg_risk * 100, 1),
            'high_risk_pct': round((high_risk_count / total) * 100, 1),
            'medium_risk_pct': round((medium_risk_count / total) * 100, 1),
            'low_risk_pct': round((low_risk_count / total) * 100, 1)
        }

    def get_all_charts(self):
        """Return all chart data"""
        return {
            'risk_distribution': {
                'data': [self.get_risk_distribution()],
                'layout': {
                    'title': 'Distribusi Level Risiko Attrition',
                    'xaxis_title': 'Tingkat Risiko',
                    'yaxis_title': 'Jumlah Karyawan',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                }
            },
            'risk_by_department': {
                'data': [self.get_risk_by_department()],
                'layout': {
                    'title': 'Rata-rata Risiko Attrition per Departemen',
                    'xaxis_title': 'Departemen',
                    'yaxis_title': 'Rata-rata Risiko',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                }
            },
            'risk_by_age': {
                'data': [self.get_risk_by_age()],
                'layout': {
                    'title': 'Tren Risiko Attrition by Kelompok Usia',
                    'xaxis_title': 'Kelompok Usia',
                    'yaxis_title': 'Rata-rata Risiko Attrition',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                }
            },
            'satisfaction_heatmap': {
                'data': [self.get_satisfaction_heatmap()],
                'layout': {
                    'title': 'Heatmap: Kepuasan vs Risiko Attrition',
                    'xaxis_title': 'Job Satisfaction',
                    'yaxis_title': 'Environment Satisfaction',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'height': 500,
                }
            },
            'tenure_scatter': {
                'data': self.get_tenure_scatter(),
                'layout': {
                    'title': 'Tenure vs Risiko Attrition (ukuran = Pendapatan)',
                    'xaxis_title': 'Tahun di Perusahaan',
                    'yaxis_title': 'Probabilitas Attrition',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'height': 500,
                }
            }
        }
