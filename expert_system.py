"""
Expert System untuk Attrition Prediction
Rule-based penjelasan sebab-akibat untuk HR decision-making
"""

class AttritionExpertSystem:
    """
    Hybrid approach: Model (60%) + Rule-Base (40%)
    Output: Risk score + Top drivers dengan penjelasan detail
    """

    def __init__(self):
        # Mapping untuk penjelasan dalam Bahasa Indonesia
        self.driver_explanations = {
            'job_satisfaction': {
                'label': 'Kepuasan Kerja',
                'reason_high': 'Karyawan tidak merasa valued, challenged, atau engaged dengan pekerjaan',
                'reason_medium': 'Karyawan mungkin membutuhkan tantangan tambahan atau engagement lebih dalam pekerjaannya',
                'reason_low': 'Ada indikasi kepuasan kerja belum optimal, namun belum menunjukkan risiko attrition yang signifikan',
                'actions_high': [
                    'Lakukan career discussion untuk understand aspirasi',
                    'Assess skill gaps dan provide development opportunities',
                    'Review job enrichment atau role redesign',
                    'Increase recognition & feedback frequency'
                ],
                'actions_medium': [
                    'Berikan feedback dan recognition lebih rutin',
                    'Diskusikan peluang pengembangan atau pelatihan',
                    'Identifikasi area pekerjaan yang bisa dibuat lebih menarik'
                ],
                'actions_low': [
                    'Berikan feedback dan recognition lebih rutin',
                    'Diskusikan peluang pengembangan atau pelatihan',
                    'Identifikasi area pekerjaan yang bisa dibuat lebih menarik'
                ]
            },
            'overtime_burnout': {
                'label': 'Beban Kerja Tinggi + Engagement Rendah',
                'reason_high': 'Kombinasi kerja berlebihan dengan engagement rendah → burnout risk tinggi',
                'reason_medium': 'Ada indikasi keseimbangan kerja dan kepuasan yang perlu dimonitor',
                'reason_low': 'Kombinasi faktor yang perlu di-monitor, namun belum menunjukkan risiko urgent',
                'actions_high': [
                    'Immediate workload review dan team capacity assessment',
                    'Identify pending projects untuk redistribution',
                    'Discuss flexible arrangements atau temporary support',
                    'Monitor health & wellbeing indicators'
                ],
                'actions_medium': [
                    'Monitor workload secara berkala',
                    'Tawarkan fleksibilitas kerja jika memungkinkan',
                    'Diskusikan well-being dan work-life balance'
                ],
                'actions_low': [
                    'Monitor workload dan working hours',
                    'Tawarkan fleksibilitas kerja jika memungkinkan',
                    'Lakukan check-in rutin terkait well-being'
                ]
            },
            'no_promotion': {
                'label': 'Stagnasi Karir (Tanpa Promosi > 3 Tahun)',
                'reason_high': 'Karyawan merasa stuck tanpa growth path → mencari opportunity lain',
                'reason_medium': 'Karyawan mungkin memerlukan clarity tentang career path dan growth opportunities',
                'reason_low': 'Ada indikasi stagnasi karir, namun belum menunjukkan risiko attrition yang signifikan',
                'actions_high': [
                    'Create succession planning & clear career ladder',
                    'Offer lateral moves atau project leadership opportunities',
                    'Discuss realistic timeline untuk next promotion',
                    'Provide skill development yang relevant untuk next level'
                ],
                'actions_medium': [
                    'Diskusikan career path dan opportunities',
                    'Identifikasi skill gaps untuk next level',
                    'Tawarkan project leadership atau mentoring role'
                ],
                'actions_low': [
                    'Diskusikan career path dan ambisi',
                    'Berikan visibility tentang jalur promosi di masa depan',
                    'Tawarkan development opportunities'
                ]
            },
            'env_satisfaction': {
                'label': 'Kepuasan Lingkungan Kerja',
                'reason_high': 'Toxic culture, poor management, atau team conflict → unsuitable workplace',
                'reason_medium': 'Ada concern tentang lingkungan kerja yang perlu diklarifikasi dan ditangani',
                'reason_low': 'Ada indikasi lingkungan kerja belum optimal, namun belum menunjukkan risiko urgent',
                'actions_high': [
                    'Deep dive conversation tentang specific issues',
                    'Cross-team assessment untuk culture health',
                    'Leadership coaching jika issue adalah manager',
                    'Consider team change atau mentorship pairing'
                ],
                'actions_medium': [
                    'Diskusikan concern spesifik tentang lingkungan kerja',
                    'Facilitate team building atau komunikasi yang lebih baik',
                    'Monitor team dynamics'
                ],
                'actions_low': [
                    'Lakukan regular check-in tentang lingkungan kerja',
                    'Monitor team dynamics dan relationships',
                    'Provide support jika diperlukan'
                ]
            },
            'work_life_balance': {
                'label': 'Keseimbangan Kerja-Hidup',
                'reason_high': 'Karyawan mengalami stress dan kelelahan signifikan. Burnout imminent jika tidak segera ditangani.',
                'reason_medium': 'Ada indikasi keseimbangan kerja-hidup yang perlu ditingkatkan. Monitor untuk prevent escalation.',
                'reason_low': 'Ada indikasi keseimbangan kerja-hidup belum optimal. Monitor ongoing untuk maintain engagement.',
                'actions_high': [
                    'Review working hours dan workload distribution segera',
                    'Normalize remote/hybrid arrangements immediately',
                    'Diskusikan flexible schedule atau sabbatical possibilities',
                    'Connect dengan wellness programs dan resources'
                ],
                'actions_medium': [
                    'Monitor workload dan working hours secara berkala',
                    'Tawarkan fleksibilitas kerja jika memungkinkan',
                    'Lakukan regular check-in tentang well-being'
                ],
                'actions_low': [
                    'Monitor workload dan working hours',
                    'Tawarkan fleksibilitas kerja jika diperlukan',
                    'Schedule regular well-being check-ins'
                ]
            },
            'low_income': {
                'label': 'Ketidaksesuaian Kompensasi (Signifikan)',
                'reason_high': 'Kompensasi jauh di bawah standar pasar untuk level dan pengalaman. Ini merupakan faktor struktural yang signifikan untuk risiko attrition.',
                'reason_medium': 'Kompensasi berada di bawah ekspektasi untuk peran dan pengalaman. Potensi risiko retensi jangka panjang yang perlu diperhatikan.',
                'reason_low': 'Kompensasi mungkin di bawah standar pasar untuk level dan tenure. Meskipun probabilitas attrition rendah, ini tetap menjadi faktor perhatian untuk equity internal dan retensi jangka panjang.',
                'actions_high': [
                    'Lakukan market salary benchmark segera',
                    'Pertimbangkan salary adjustment fast-track jika qualified',
                    'Jelaskan compensation roadmap dan promotion timeline dengan jelas',
                    'Pertimbangkan bonus atau stock options sebagai retention strategy'
                ],
                'actions_medium': [
                    'Lakukan market salary benchmarking dalam 30 hari',
                    'Review kompensasi relatif terhadap rekan sejawat',
                    'Komunikasikan jalur kenaikan kompensasi yang jelas',
                    'Assess kebutuhan adjustments'
                ],
                'actions_low': [
                    'Lakukan market salary benchmarking',
                    'Validasi alignment kompensasi dengan role, level, dan tenure',
                    'Monitor perkembangan kompensasi di review berikutnya',
                    'Pastikan karyawan memahami struktur benefit lengkap'
                ]
            },
            'role_stagnation': {
                'label': 'Stagnasi Role',
                'reason_high': 'Monotony, skill underutilization, atau boredom → seeking new challenge',
                'reason_medium': 'Karyawan mungkin memerlukan tantangan baru atau perubahan untuk maintain engagement',
                'reason_low': 'Ada indikasi tenaga kerja di role yang sama cukup lama, pertahankan engagement',
                'actions_high': [
                    'Explore cross-functional projects',
                    'Offer training untuk expand capability',
                    'Discuss rotation options atau acting role',
                    'Identify mentoring responsibilities'
                ],
                'actions_medium': [
                    'Tawarkan project lintas fungsi',
                    'Diskusikan training atau skill development',
                    'Identifikasi mentoring atau leadership opportunities'
                ],
                'actions_low': [
                    'Tawarkan project atau tanggung jawab baru',
                    'Diskusikan development opportunities',
                    'Maintain engagement melalui growth'
                ]
            },
            'relationship_satisfaction': {
                'label': 'Kepuasan Hubungan Kerja',
                'reason_high': 'Poor team dynamics atau manager relationship → uncomfortable work environment',
                'reason_medium': 'Ada indikasi concern tentang hubungan kerja yang perlu ditangani',
                'reason_low': 'Ada indikasi hubungan kerja belum optimal, perlu monitoring ringan',
                'actions_high': [
                    'Mediation atau team building sessions',
                    'Manager coaching on communication/feedback',
                    'Consider team restructuring jika needed',
                    'Provide coaching untuk employee interpersonal skills'
                ],
                'actions_medium': [
                    'Facilitate komunikasi yang lebih baik dengan manager/team',
                    'Tawarkan training soft skills atau team building',
                    'Monitor perkembangan relationship'
                ],
                'actions_low': [
                    'Monitor team dynamics dan relationships',
                    'Facilitate komunikasi yang konstruktif',
                    'Maintain supportive work environment'
                ]
            }
        }

    def calculate_rule_score(self, data):
        """
        Hitung rule-based score (0-100)
        Data dict harus contain: age, job_satisfaction, overtime, years_since_promotion,
                               env_satisfaction, work_life_balance, monthly_income,
                               years_in_role, relationship_satisfaction, job_level, dept_avg_income
        """
        score = 0
        active_drivers = []

        # TIER 1: CRITICAL Red Flags (40+ points each)
        if data.get('job_satisfaction') == 1:
            score += 40
            active_drivers.append(('job_satisfaction', 40))

        # Overtime Burnout: OverTime + JobSat <= 2
        if data.get('overtime') == 'Yes' and data.get('job_satisfaction', 5) <= 2:
            score += 35
            active_drivers.append(('overtime_burnout', 35))

        # No promotion > 3 years
        if data.get('years_since_promotion', 0) > 3:
            score += 25
            active_drivers.append(('no_promotion', 25))

        # Environment satisfaction = 1
        if data.get('env_satisfaction') == 1:
            score += 25
            active_drivers.append(('env_satisfaction', 25))

        # Work-life balance very poor
        if data.get('work_life_balance') <= 1:
            score += 20
            active_drivers.append(('work_life_balance', 20))

        # TIER 2: WARNING Signs (10-20 points)
        # Income below department average
        monthly_income = data.get('monthly_income', 0)
        dept_avg = data.get('dept_avg_income', monthly_income)
        if monthly_income > 0 and monthly_income < dept_avg * 0.8:  # 20% below avg
            score += 15
            active_drivers.append(('low_income', 15))

        # Role stagnation > 5 years
        if data.get('years_in_role', 0) > 5 and data.get('years_since_promotion', 0) > 2:
            score += 15
            active_drivers.append(('role_stagnation', 15))

        # Job satisfaction <= 2
        if data.get('job_satisfaction', 5) in [1, 2]:
            if ('job_satisfaction', 40) not in active_drivers:  # Jangan double count
                score += 10
                if 'job_satisfaction' not in [d[0] for d in active_drivers]:
                    active_drivers.append(('job_satisfaction', 10))

        # Relationship satisfaction <= 2
        if data.get('relationship_satisfaction', 5) <= 2:
            score += 10
            active_drivers.append(('relationship_satisfaction', 10))

        # TIER 3: Contextual Factors (5-10 points)
        age = data.get('age', 30)
        if age < 25 and monthly_income < 4000:
            score += 10

        # Job hopper (many companies worked)
        if data.get('num_companies_worked', 0) > 5:
            score += 8

        # Cap at 100
        score = min(score, 100)

        return score, active_drivers

    def get_top_drivers(self, active_drivers, probability, risk_level):
        """
        Return top 3 drivers dengan penjelasan & aksi (differentiated by risk level)
        """
        # Sort by score descending
        sorted_drivers = sorted(active_drivers, key=lambda x: x[1], reverse=True)

        top_3 = []
        for i, (driver_key, driver_score) in enumerate(sorted_drivers[:3], 1):
            if driver_key in self.driver_explanations:
                info = self.driver_explanations[driver_key].copy()

                # Pilih reason & actions based on risk level
                if risk_level == "TINGGI":
                    reason_key = 'reason_high'
                    actions_key = 'actions_high'
                elif risk_level == "SEDANG":
                    reason_key = 'reason_medium'
                    actions_key = 'actions_medium'
                else:  # RENDAH
                    reason_key = 'reason_low'
                    actions_key = 'actions_low'

                info['reason'] = info.get(reason_key, info.get('reason', ''))
                info['actions'] = info.get(actions_key, info.get('actions', []))
                info['score'] = driver_score
                info['rank'] = i
                top_3.append(info)

        return top_3

    def calculate_hybrid_risk(self, model_probability, rule_score):
        """
        Hybrid: 60% Model + 40% Rule-Base
        Return: risk_percentage, risk_level, drivers
        """
        # Normalize rule_score (0-100) to probability (0-1)
        rule_probability = rule_score / 100.0

        # Weighted combination
        hybrid_prob = (0.6 * model_probability) + (0.4 * rule_probability)

        # Risk level
        if hybrid_prob > 0.60:
            risk_level = "TINGGI"
            emoji = "🔴"
        elif hybrid_prob > 0.30:
            risk_level = "SEDANG"
            emoji = "🟡"
        else:
            risk_level = "RENDAH"
            emoji = "🟢"

        return hybrid_prob, risk_level, emoji

    def get_recommendations(self, risk_level, top_drivers):
        """
        Berikan rekomendasi aksi berdasarkan risk level & drivers
        Risk level determines tone dan urgency
        """
        if risk_level == "TINGGI":
            priority = "SEGERA (minggu ini)"
            general = "⚠️ INTERVENSI SEGERA DIPERLUKAN: Karyawan dalam risiko tinggi untuk keluar. Prioritas: career development, compensation review, dan workload assessment."

        elif risk_level == "SEDANG":
            priority = "DALAM 2 MINGGU"
            general = "🟡 MONITOR & ENGAGE: Jadwalkan pembicaraan career development dan verifikasi kepuasan di berbagai dimensi. Intervensi proaktif direkomendasikan dalam 2 minggu ke depan."

        else:  # RENDAH
            priority = "RUTIN (Quarterly)"
            general = "🟢 PERTAHANKAN ENGAGEMENT: Karyawan menunjukkan risiko attrition yang rendah. Tidak diperlukan intervensi segera, namun beberapa faktor tetap perlu dipantau untuk menjaga engagement dan retensi jangka panjang."

        # Collect actions dari top drivers dengan limit berdasarkan risk level
        actions = []
        for driver in top_drivers:
            for action in driver.get('actions', []):
                actions.append(action)

        # Limit actions based on risk level
        if risk_level == "TINGGI":
            specific_actions = actions[:6]  # Up to 6 actions for high risk
        elif risk_level == "SEDANG":
            specific_actions = actions[:5]  # Up to 5 actions for medium risk
        else:  # RENDAH
            specific_actions = actions[:4]  # Up to 4 actions for low risk

        return {
            'priority': priority,
            'general': general,
            'specific_actions': specific_actions
        }

    def predict(self, model_probability, input_data):
        """
        Full ML dengan Explainability + Additional Insights
        Layer 1: ML Probability (100% model)
        Layer 2: Key Drivers (dari top_drivers)
        Layer 3: Additional Insights (conditional flags)
        """
        # Determine risk level BERDASARKAN ML probability
        if model_probability > 0.60:
            risk_level = "TINGGI"
            emoji = "🔴"
        elif model_probability > 0.30:
            risk_level = "SEDANG"
            emoji = "🟡"
        else:
            risk_level = "RENDAH"
            emoji = "🟢"

        # Calculate rule-based score untuk drivers
        rule_score, active_drivers = self.calculate_rule_score(input_data)

        # LAYER 2: Get top drivers (sorted by severity)
        top_drivers = self.get_top_drivers(active_drivers, model_probability, risk_level)

        # LAYER 3: Generate Additional Insights (rule-based flags)
        additional_insights = self._generate_additional_insights(input_data, model_probability, risk_level)

        # LAYER 1: Get recommendations
        recommendations = self.get_recommendations(risk_level, top_drivers)

        # Detect contradiction dengan logic lebih ketat
        contradiction_warning = self._detect_contradiction(model_probability, rule_score, input_data, risk_level)

        return {
            'ml_probability': model_probability,
            'risk_level': risk_level,
            'emoji': emoji,
            'top_drivers': top_drivers,
            'recommendations': recommendations,
            'contradiction_warning': contradiction_warning,
            'additional_insights': additional_insights,
            'rule_score': rule_score
        }

    def _detect_contradiction(self, ml_prob, rule_score, input_data, risk_level):
        """
        Detect contradiction yang lebih akurat + tegas
        Mempertimbangkan severity level
        """
        contradiction = None

        # Case 1: Low ML prob tapi SIGNIFICANT HR risk (rule_score > 30)
        if ml_prob < 0.30 and rule_score > 30:
            # Check if ada compensation issue yang signifikan
            monthly_income = input_data.get('monthly_income', 5000)
            job_level = input_data.get('job_level', 2)
            years_at_company = input_data.get('years_at_company', 5)

            # Red flag: Income sangat rendah untuk senior tenure
            if monthly_income < 3000 and years_at_company > 10:
                contradiction = "⚠️ ANOMALI SIGNIFIKAN: Meskipun model memprediksi risiko rendah, terdapat ketidaksesuaian kompensasi yang jelas untuk tenure dan pengalaman karyawan. Ini menunjukkan potensi risiko retensi jangka panjang yang perlu segera ditangani."
            else:
                contradiction = "⚠️ FAKTOR LATEN TERDETEKSI: Meskipun probabilitas attrition rendah menurut model, beberapa faktor HR signifikan terdeteksi yang dapat mempengaruhi retensi dalam jangka panjang. Monitoring dan assessment tambahan direkomendasikan."

        return contradiction

    def _generate_additional_insights(self, input_data, ml_prob, risk_level):
        """
        Generate insights tambahan berdasarkan kondisi khusus
        Berlaku untuk semua case
        """
        insights = []

        monthly_income = input_data.get('monthly_income', 5000)
        job_level = input_data.get('job_level', 2)
        years_at_company = input_data.get('years_at_company', 5)
        years_in_role = input_data.get('years_in_role', 3)
        job_satisfaction = input_data.get('job_satisfaction', 3)
        work_life_balance = input_data.get('work_life_balance', 3)
        overtime = input_data.get('overtime', 'No')
        age = input_data.get('age', 35)
        years_since_promotion = input_data.get('years_since_promotion', 2)

        # FLAG 1: Compensation Mismatch (berlaku ke semua)
        if monthly_income < 3000 and years_at_company > 5:
            insights.append({
                'category': 'Ketidaksesuaian Kompensasi',
                'severity': 'high',
                'description': 'Pendapatan jauh di bawah ekspektasi untuk tenure dan pengalaman. Risiko retensi jangka panjang.'
            })
        elif monthly_income < 4000 and job_level >= 3:
            insights.append({
                'category': 'Ketidaksesuaian Kompensasi',
                'severity': 'medium',
                'description': 'Kompensasi mungkin di bawah standar untuk level seniority. Equity check diperlukan.'
            })

        # FLAG 2: Stagnation Risk (berlaku ke semua)
        if years_in_role > 5 and years_since_promotion > 3:
            insights.append({
                'category': 'Stagnasi Karir',
                'severity': 'high',
                'description': 'Tenure panjang di role/posisi sama tanpa promosi. Potensi disengagement.'
            })
        elif years_since_promotion > 4:
            insights.append({
                'category': 'Stagnasi Karir',
                'severity': 'medium',
                'description': 'Lama tanpa promosi. Pertimbangkan career development opportunities.'
            })

        # FLAG 3: Burnout Risk (berlaku ke semua)
        if overtime == 'Yes' and (work_life_balance <= 2 or job_satisfaction <= 2):
            insights.append({
                'category': 'Burnout Risk',
                'severity': 'high',
                'description': 'Kombinasi overtime + rendah WLB/satisfaction. Immediate intervention diperlukan.'
            })
        elif overtime == 'Yes' and work_life_balance <= 2:
            insights.append({
                'category': 'Burnout Risk',
                'severity': 'medium',
                'description': 'Overtime dengan WLB rendah. Monitor kesehatan dan engagement.'
            })

        # FLAG 4: Engagement Risk (berlaku ke semua)
        if job_satisfaction <= 2 and job_level <= 2:
            insights.append({
                'category': 'Risiko Disengagement',
                'severity': 'high',
                'description': 'Low satisfaction pada level junior. Career development urgent.'
            })
        elif job_satisfaction <= 2:
            insights.append({
                'category': 'Risiko Disengagement',
                'severity': 'medium',
                'description': 'Kepuasan kerja rendah. Diskusi career development diperlukan.'
            })

        # FLAG 5: Early-stage adjustment (berlaku ke semua)
        if years_at_company <= 1 and ml_prob > 0.30:
            insights.append({
                'category': 'Early-Stage Review',
                'severity': 'low',
                'description': 'Karyawan baru dengan medium risk probability. Ini mungkin normal adjustment period. Monitor saja.'
            })

        return insights
