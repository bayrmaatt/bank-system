import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Банкны Систем",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin-top: 2rem;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #28a745;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #ffc107;
}
.danger-box {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

if 'qa_question' not in st.session_state:
    st.session_state.qa_question = ''

@st.cache_resource
def load_banking_models():
    """Банкны AI моделуудыг ачаалах"""
    models = {}
    
    try:
        models['education_encoder'] = joblib.load('models/education_encoder.pkl')
        models['marital_encoder'] = joblib.load('models/marital_encoder.pkl')
        models['location_encoder'] = joblib.load('models/location_encoder.pkl')
        models['txn_type_encoder'] = joblib.load('models/txn_type_encoder.pkl')
        models['merchant_encoder'] = joblib.load('models/merchant_encoder.pkl')
        
        models['loan_model'] = load_model('models/loan_model.h5')
        models['loan_scaler'] = joblib.load('models/loan_scaler.pkl')
        models['loan_feature_columns'] = joblib.load('models/loan_feature_columns.pkl')
        
        models['kmeans'] = joblib.load('models/kmeans_model.pkl')
        models['cluster_scaler'] = joblib.load('models/cluster_scaler.pkl')
        models['clustering_features'] = joblib.load('models/clustering_features.pkl')
        
        models['rf_fraud'] = joblib.load('models/rf_fraud.pkl')
        models['gb_fraud'] = joblib.load('models/gb_fraud.pkl')
        models['fraud_feature_columns'] = joblib.load('models/fraud_feature_columns.pkl')
        
        models['qa_vectorizer'] = joblib.load('models/qa_vectorizer.pkl')
        models['qa_answers'] = joblib.load('models/qa_answers.pkl')
        models['question_vectors'] = joblib.load('models/question_vectors.pkl')
        
        try:
            models['sample_customers'] = pd.read_csv('data/sample_customers.csv')
            models['sample_transactions'] = pd.read_csv('data/sample_transactions.csv')
        except:
            models['sample_customers'] = None
            models['sample_transactions'] = None
        
        st.success("Бүх модель амжилттай ачаалагдлаа!")
        return models
    
    except FileNotFoundError as e:
        st.error(f"Модель файл олдсонгүй: {e}")
        st.error("Эхлээд Jupyter notebook кодыг ажиллуулж моделуудыг үүсгэнэ үү.")
        return None
    except Exception as e:
        st.error(f"Модель ачаалахад алдаа: {e}")
        return None

def predict_loan_approval(models, customer_data):
    """Зээлийн зөвшөөрөл таамаглах"""
    try:
        features = np.array([
            customer_data['age'],
            customer_data['income'],
            customer_data['employment_years'],
            customer_data['loan_amount'],
            customer_data['loan_term'],
            customer_data['credit_history_months'],
            customer_data['existing_loans'],
            customer_data['collateral_value'],
            customer_data['bank_relationship_years'],
            customer_data['monthly_expense'],
            customer_data['has_savings'],
            customer_data['credit_score'],
            customer_data['loan_amount'] / customer_data['income'],
            customer_data['collateral_value'] / customer_data['loan_amount'],
            customer_data['age'] * customer_data['income'] / 1000000,
            customer_data['loan_amount'] / customer_data['loan_term'],
            (customer_data['loan_amount'] / customer_data['loan_term']) / (customer_data['income'] / 12),
            models['education_encoder'].transform([customer_data['education']])[0],
            models['marital_encoder'].transform([customer_data['marital_status']])[0],
            models['location_encoder'].transform([customer_data['location']])[0]
        ]).reshape(1, -1)
        
        features_scaled = models['loan_scaler'].transform(features)
        probability = models['loan_model'].predict(features_scaled)[0][0]
        
        return {
            'probability': float(probability),
            'decision': 'Зөвшөөрөх' if probability > 0.5 else 'Татгалзах',
            'risk_level': 'Бага' if probability > 0.7 else 'Дунд' if probability > 0.3 else 'Өндөр'
        }
    except Exception as e:
        return {'error': str(e)}

def predict_customer_segment(models, customer_data):
    """Харилцагчийн сегмент таамаглах"""
    try:
        age = customer_data['age']
        income = customer_data['income']
        employment_years = customer_data['employment_years']
        bank_relationship_years = customer_data['bank_relationship_years']
        existing_loans = customer_data['existing_loans']
        monthly_expense = customer_data['monthly_expense']
        credit_score = customer_data['credit_score']
        
        if income >= 5000000 and credit_score >= 750 and bank_relationship_years >= 5:
            segment_id = 3  
            segment_name = 'VIP харилцагч'
        elif income >= 3000000 and credit_score >= 700:
            segment_id = 2 
            segment_name = 'Premium харилцагч'
        elif income >= 4000000 and bank_relationship_years <= 2:
            segment_id = 5 
            segment_name = 'Өндөр орлоготай харилцагч'
        elif credit_score < 500 or existing_loans > 3 or monthly_expense > income * 0.8:
            segment_id = 4 
            segment_name = 'Эрсдэлтэй харилцагч'
        elif bank_relationship_years <= 1:
            segment_id = 0 
            segment_name = 'Шинэ харилцагч'
        else:
            segment_id = 1 
            segment_name = 'Стандарт харилцагч'
        
        return {
            'segment_id': segment_id,
            'segment_name': segment_name,
        }
    except Exception as e:
        return {'error': str(e)}

def predict_fraud_risk(models, transaction_data):
    """Fraud эрсдэл таамаглах"""
    try:
        features = np.array([
            transaction_data['amount'],
            transaction_data['time_hour'],
            transaction_data['day_of_week'],
            1 if transaction_data['day_of_week'] in [6, 7] else 0,
            1 if transaction_data['time_hour'] < 6 or transaction_data['time_hour'] > 22 else 0,
            np.log1p(transaction_data['amount']),
            (transaction_data['amount'] - 50000) / 100000,
            transaction_data['is_international'],
            models['txn_type_encoder'].transform([transaction_data['transaction_type']])[0],
            models['location_encoder'].transform([transaction_data['location']])[0],
            models['merchant_encoder'].transform([transaction_data['merchant_category']])[0]
        ]).reshape(1, -1)
        
        rf_prob = models['rf_fraud'].predict_proba(features)[0][1]
        gb_prob = models['gb_fraud'].predict_proba(features)[0][1]
        avg_prob = (rf_prob + gb_prob) / 2
        
        return {
            'fraud_probability': float(avg_prob),
            'risk_level': 'Өндөр' if avg_prob > 0.7 else 'Дунд' if avg_prob > 0.3 else 'Бага',
            'recommendation': 'Блоклох' if avg_prob > 0.7 else 'Шалгах' if avg_prob > 0.3 else 'Зөвшөөрөх'
        }
    except Exception as e:
        return {'error': str(e)}

def answer_question(models, question):
    """Q&A систем"""
    try:
        question_vector = models['qa_vectorizer'].transform([question.lower()])
        similarities = cosine_similarity(question_vector, models['question_vectors'])[0]
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        
        return {
            'answer': models['qa_answers'][best_match_idx],
            'confidence': float(confidence),
            'reliable': confidence > 0.3
        }
    except Exception as e:
        return {'error': str(e)}

def show_home_page():
    """Нүүр хуудас"""
    st.markdown("### Банкны AI системийн боломжууд")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Зээлийн шинжилгээ
        - Зээлийн эрсдэлийн үнэлгээ
        - Автомат зөвшөөрөл
        - Нарийвчилсан таамаглал
        
        #### Аюулгүй байдал
        - Real-time fraud илрүүлэх
        - Сэжигтэй гүйлгээ
        """)
    
    with col2:
        st.markdown("""
        #### Харилцагчийн сегментчилэл
        - 6 өөр сегментэд ангилах
        - Хувийн санал
        - Зорилтот маркетинг
        
        #### Асуулт хариулт
        - 24/7 автомат дэмжлэг
        - Монгол хэл дээр
        - Банкны бүтээгдэхүүний мэдээлэл
        """)
    
    st.markdown("### Систем")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Зээлийн модель", "Онлайн", "Идэвхтэй")
    with col2:
        st.metric("Fraud модель", "Онлайн", "Идэвхтэй")
    with col3:
        st.metric("Харилцагчийн сегмент", "6 сегмент", "Бэлэн")
    with col4:
        st.metric("Q&A систем", "Онлайн", "Бэлэн")

def show_loan_analysis(models):
    """Зээлийн шинжилгээ хуудас"""
    st.markdown('<h2 class="sub-header">Зээлийн шинжилгээ</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Харилцагчийн мэдээлэл")
        
        age = st.slider("Нас", 18, 70, 35)
        income = st.number_input("Сарын орлого (₮)", min_value=0, value=1500000, step=50000)
        employment_years = st.slider("Ажлын туршлага (жил)", 0, 30, 5)
        
        education = st.selectbox("Боловсрол", ['Дунд', 'Дээд', 'Магистр', 'Доктор'])
        marital_status = st.selectbox("Гэрлэлтийн байдал", ['Гэрлэсэн', 'Ганц', 'Салсан'])
        location = st.selectbox("Хот/аймаг", ['Улаанбаатар', 'Дархан', 'Эрдэнэт', 'Чойр', 'Мөрөн'])
        
        st.markdown("#### Зээлийн мэдээлэл")
        
        loan_amount = st.number_input("Зээлийн дүн (₮)", min_value=0, value=30000000, step=1000000)
        loan_term = st.selectbox("Хугацаа (сар)", [12, 24, 36, 48, 60, 72])
        collateral_value = st.number_input("Барьцааны үнэ (₮)", min_value=0, value=50000000, step=1000000)
        
        credit_score = st.slider("Кредитийн оноо", 300, 850, 650)
        existing_loans = st.slider("Одоо байгаа зээлийн тоо", 0, 5, 1)
        credit_history_months = st.slider("Кредитийн түүх (сар)", 0, 120, 24)
        
        bank_relationship_years = st.slider("Банктай харилцсан жил", 0, 20, 3)
        monthly_expense = st.number_input("Сарын зарлага (₮)", min_value=0, value=800000, step=50000)
        has_savings = st.checkbox("Хуримтлал бий эсэх", value=True)
    
    with col2:
        if st.button(" Шинжилгээ хийх", type="primary"):
            customer_data = {
                'age': age,
                'income': income,
                'employment_years': employment_years,
                'education': education,
                'marital_status': marital_status,
                'location': location,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'credit_history_months': credit_history_months,
                'existing_loans': existing_loans,
                'collateral_value': collateral_value,
                'bank_relationship_years': bank_relationship_years,
                'monthly_expense': monthly_expense,
                'has_savings': 1 if has_savings else 0,
                'credit_score': credit_score
            }
            
            result = predict_loan_approval(models, customer_data)
            
            if 'error' in result:
                st.error(f" Алдаа: {result['error']}")
            else:
                probability = result['probability'] * 100
                
                if result['decision'] == 'Зөвшөөрөх':
                    st.markdown(f"""
                    <div class="success-box">
                        <h3> Зээл зөвшөөрөгдөнө</h3>
                        <p><strong>Магадлал:</strong> {probability:.1f}%</p>
                        <p><strong>Эрсдлийн түвшин:</strong> {result['risk_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3> Зээл татгалзагдана</h3>
                        <p><strong>Магадлал:</strong> {probability:.1f}%</p>
                        <p><strong>Эрсдлийн түвшин:</strong> {result['risk_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Зөвшөөрөлийн магадлал (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                

def show_customer_segmentation(models):
    """Харилцагчийн сегментчилэл хуудас"""
    st.markdown('<h2 class="sub-header">Харилцагчийн сегментчилэл</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Харилцагчийн мэдээлэл")
        
        age = st.slider("Нас", 18, 70, 35, key="seg_age")
        income = st.number_input("Сарын орлого (₮)", min_value=0, value=1500000, step=50000, key="seg_income")
        employment_years = st.slider("Ажлын туршлага (жил)", 0, 30, 5, key="seg_employment")
        bank_relationship_years = st.slider("Банктай харилцсан жил", 0, 20, 3, key="seg_bank_rel")
        existing_loans = st.slider("Одоо байгаа зээлийн тоо", 0, 5, 1, key="seg_loans")
        monthly_expense = st.number_input("Сарын зарлага (₮)", min_value=0, value=800000, step=50000, key="seg_expense")
        credit_score = st.slider("Кредитийн оноо", 300, 850, 650, key="seg_credit")
    
    with col2:
        if st.button(" Сегмент тодорхойлох", type="primary"):
            customer_data = {
                'age': age,
                'income': income,
                'employment_years': employment_years,
                'bank_relationship_years': bank_relationship_years,
                'existing_loans': existing_loans,
                'monthly_expense': monthly_expense,
                'credit_score': credit_score
            }
            
            result = predict_customer_segment(models, customer_data)
            
            if 'error' in result:
                st.error(f" Алдаа: {result['error']}")
            else:
                segment_colors = {
                    'Шинэ харилцагч': '#FF6B6B',
                    'Стандарт харилцагч': '#4ECDC4',
                    'Premium харилцагч': '#45B7D1',
                    'VIP харилцагч': '#FFA726',
                    'Эрсдэлтэй харилцагч': '#EF5350',
                    'Өндөр орлоготой харилцагч': '#66BB6A'
                }
                
                segment_name = result['segment_name']
                color = segment_colors.get(segment_name, '#666666')
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 2rem; border-radius: 10px; border-left: 5px solid {color};">
                    <h3 style="color: {color};">{segment_name}</h3>
                    <p><strong>Сегментийн ID:</strong> {result['segment_id']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                segment_descriptions = {
                    'Шинэ харилцагч': {
                        'desc': 'Банктай дөнгөж харилцаж эхэлсэн харилцагч',
                        'features': ['Бага банкны түүх', 'Стандарт үйлчилгээ', 'Анхны санал хүлээн авсан'],
                        'recommendations': ['Welcome packet санал болгох', 'Анхны зээлийн хөнгөлөлт', 'Санхүүгийн зөвлөгөө өгөх']
                    },
                    'Стандарт харилцагч': {
                        'desc': 'Банкны үндсэн харилцагч бүлэг',
                        'features': ['Дундаж орлого', 'Тогтмол банкны үйлчилгээ', 'Стандарт зээлийн түүх'],
                        'recommendations': ['Стандарт үйлчилгээний packet', 'Жилийн хөнгөлөлт', 'Урамшуулалын программ']
                    },
                    'Premium харилцагч': {
                        'desc': 'Өндөр орлоготой, чанартай харилцагч',
                        'features': ['Өндөр орлого', 'Сайн кредитийн түүх', 'Олон төрлийн үйлчилгээ ашигладаг'],
                        'recommendations': ['Premium банкны үйлчилгээ', 'Хувийн банкир', 'Хөрөнгө оруулалтын зөвлөгөө']
                    },
                    'VIP харилцагч': {
                        'desc': 'Банкны хамгийн чухал харилцагч',
                        'features': ['Маш өндөр орлого', 'Олон жилийн түүх', 'Өргөн хүрээний үйлчилгээ'],
                        'recommendations': ['VIP үйлчилгээ', 'Онцгой хөнгөлөлт', 'Хувийн санхүүгийн зөвлөх']
                    },
                    'Эрсдэлтэй харилцагч': {
                        'desc': 'Санхүүгийн эрсдэлтэй харилцагч',
                        'features': ['Муу кредитийн түүх', 'Тогтмол бус орлого', 'Олон зээлтэй'],
                        'recommendations': ['Илүү анхаарал хандах', 'Зээлийн хязгаарлалт', 'Санхүүгийн зөвлөгөө өгөх']
                    },
                    'Өндөр орлоготой харилцагч': {
                        'desc': 'Өндөр орлоготой боловч шинэ харилцагч',
                        'features': ['Маш өндөр орлого', 'Богино банкны түүх', 'Том хөрөнгө оруулалтын боломж'],
                        'recommendations': ['Хөрөнгө оруулалтын бүтээгдэхүүн', 'Private banking', 'Онцгой анхаарал']
                    }
                }
                
                if segment_name in segment_descriptions:
                    info = segment_descriptions[segment_name]
                    
                    st.markdown("####  Сегментийн тодорхойлолт")
                    st.write(info['desc'])
                    
                    col_feat, col_rec = st.columns(2)
                    
                    with col_feat:
                        st.markdown("**Онцлогууд:**")
                        for feature in info['features']:
                            st.write(f"• {feature}")
                    
                    with col_rec:
                        st.markdown("**Зөвлөмжүүд:**")
                        for rec in info['recommendations']:
                            st.write(f"• {rec}")
        
        if models.get('sample_customers') is not None:
            st.markdown("####  Сегментийн хуваарилалт")
            
            segment_counts = {
                'Шинэ харилцагч': 15,
                'Стандарт харилцагч': 35,
                'Premium харилцагч': 20,
                'VIP харилцагч': 10,
                'Эрсдэлтэй харилцагч': 12,
                'Өндөр орлоготой харилцагч': 8
            }
            
            segment_colors = {
                'Шинэ харилцагч': '#FF6B6B',
                'Стандарт харилцагч': '#4ECDC4',
                'Premium харилцагч': '#45B7D1',
                'VIP харилцагч': '#FFA726',
                'Эрсдэлтэй харилцагч': '#EF5350',
                'Өндөр орлоготой харилцагч': '#66BB6A'
            }
            
            fig = px.pie(
                values=list(segment_counts.values()),
                names=list(segment_counts.keys()),
                title="Харилцагчдын сегментийн хуваарилалт",
                color_discrete_map=segment_colors
            )
            st.plotly_chart(fig, use_container_width=True)

def show_fraud_detection(models):
    """Fraud илрүүлэх хуудас"""
    st.markdown('<h2 class="sub-header"> Fraud илрүүлэх систем</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("####  Гүйлгээний мэдээлэл")
        
        amount = st.number_input("Гүйлгээний дүн (₮)", min_value=0, value=500000, step=10000)
        time_hour = st.slider("Цаг (24 цагийн форматаар)", 0, 23, 14)
        day_of_week = st.selectbox("Долоо хоногийн өдөр", 
                                 [1, 2, 3, 4, 5, 6, 7], 
                                 format_func=lambda x: ['Даваа', 'Мягмар', 'Лхагва', 'Пүрэв', 'Баасан', 'Бямба', 'Ням'][x-1])
        
        transaction_type = st.selectbox("Гүйлгээний төрөл", ['Шилжүүлэг', 'Төлбөр', 'Данс цэнэглэх', 'ATM', 'Online'])
        location = st.selectbox("Байршил", ['Улаанбаатар', 'Дархан', 'Эрдэнэт', 'Чойр', 'Мөрөн'])
        merchant_category = st.selectbox("Худалдаачийн ангилал", ['Супермаркет', 'Шатахуун', 'Кафе', 'Онлайн', 'ATM', 'Банк'])
        
        is_international = st.checkbox("Олон улсын гүйлгээ эсэх")
    
    with col2:
        if st.button(" Fraud шалгах", type="primary"):
            transaction_data = {
                'amount': amount,
                'time_hour': time_hour,
                'day_of_week': day_of_week,
                'transaction_type': transaction_type,
                'location': location,
                'merchant_category': merchant_category,
                'is_international': 1 if is_international else 0
            }
            
            result = predict_fraud_risk(models, transaction_data)
            
            if 'error' in result:
                st.error(f" Алдаа: {result['error']}")
            else:
                fraud_prob = result['fraud_probability'] * 100
                risk_level = result['risk_level']
                recommendation = result['recommendation']
                
                if risk_level == 'Өндөр':
                    box_class = "danger-box"
                elif risk_level == 'Дунд':
                    box_class = "warning-box"
                else:
                    box_class = "success-box"                
                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{recommendation}</h3>
                    <p><strong>Fraud магадлал:</strong> {fraud_prob:.1f}%</p>
                    <p><strong>Эрсдлийн түвшин:</strong> {risk_level}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fraud_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud магадлал (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "darkred", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("####  Эрсдлийн хүчин зүйлс")
                
                risk_factors = []
                
                if amount > 1000000:
                    risk_factors.append("Өндөр дүнтэй гүйлгээ")
                
                if time_hour < 6 or time_hour > 22:
                    risk_factors.append("Оройн цагийн гүйлгээ")
                
                if day_of_week in [6, 7]:
                    risk_factors.append("Амралтын өдрийн гүйлгээ")
                
                if is_international:
                    risk_factors.append("Олон улсын гүйлгээ")
                
                if transaction_type == 'Online':
                    risk_factors.append("Онлайн гүйлгээ")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f" {factor}")
                else:
                    st.write(" Илрүүлсэн эрсдлийн хүчин зүйл байхгүй")
        
        st.markdown("####  Fraud статистик")
        
        fraud_by_hour = np.random.beta(2, 8, 24) * 0.3  
        fraud_by_hour[0:6] *= 2
        fraud_by_hour[22:24] *= 1.5  
        
        fig = px.line(
            x=range(24), 
            y=fraud_by_hour,
            title="Эрсдэлийн түвшин цагийн хувьд",
            labels={'x': 'Цаг', 'y': 'Fraud магадлал'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_qa_system(models):
    """Q&A систем хуудас"""
    st.markdown('<h2 class="sub-header">Асуулт хариулт систем</h2>', unsafe_allow_html=True)
    
    st.markdown("####  Банкны бүтээгдэхүүний талаар асуугаарай")
    
    st.markdown("**Түгээмэл асуултууд:**")
    predefined_questions = [
        "Бизнесийн зээл авах",
        "Эмэгтэй бизнес эрхлэгчийн зээл", 
        "Хөдөө аж ахуйн зээл",
        "Жижиг бизнесийн зээл",
        "Цалингийн зээл авах",
        "ПОС зээл",
        "Органик тариаланы зээл",
        "Өрхийн хэрэгцээний зээл"
    ]
    
    col1, col2, col3, col4 = st.columns(4)
    
    for i, question in enumerate(predefined_questions):
        col = [col1, col2, col3, col4][i % 4]
        with col:
            if st.button(question, key=f"preset_{i}"):
                st.session_state.qa_question = question
    
    question = st.text_input(
        "Өөрийн асуулт бичнэ үү:",
        value=st.session_state.get('qa_question', ''),
        key="user_question"
    )
    
    if question:
        result = answer_question(models, question)
        
        if 'error' in result:
            st.error(f" Алдаа: {result['error']}")
        else:
            confidence = result['confidence'] * 100
            
            if result['reliable']:
                st.markdown(f"""
                <div class="success-box">
                    <h4> Хариулт</h4>
                    <p>{result['answer']}</p>
                    <small>Итгэлцүүр: {confidence:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4> Хариулт (бага итгэлцүүртэй)</h4>
                    <p>{result['answer']}</p>
                    <small>Итгэлцүүр: {confidence:.1f}%</small>
                    <p><em>Илүү нарийвчилсан мэдээллийг банкны ажилтангаас асууна уу.</em></p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("####  Түгээмэл асуулт хариулт")
    
    faq_data = {
        "Асуулт": [
            "Зээлийн хүү хэд вэ?",
            "Зээл авахад ямар бичиг баримт хэрэгтэй вэ?",
            "Зээлийн хугацаа хэд вэ?",
            "Онлайнаар зээл авч болох уу?",
            "Барьцаа шаардлагатай юу?"
        ],
        "Хариулт": [
            "Зээлийн хүү нь зээлийн төрлөөс хамаараад 5%-24% хооронд байна.",
            "Иргэний үнэмлэх, дансны хуулга, барьцааны бичиг баримт хэрэгтэй.",
            "Зээлийн хугацаа 12-72 сар хооронд сонгох боломжтой.",
            "Тийм, манай банкны мобайл апп болон вебсайтаар онлайн хүсэлт гаргаж болно.",
            "Зээлийн дүн, төрлөөс хамааран барьцаа шаардлагатай эсэх тодорхойлогдоно."
        ]
    }
    
    faq_df = pd.DataFrame(faq_data)
    st.dataframe(faq_df, use_container_width=True)

def show_dashboard(models):
    """Дашборд хуудас"""
    st.markdown('<h2 class="sub-header"> Ерөнхий дашборд</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Нийт харилцагч", "5,000")
    with col2:
        st.metric("Зээлийн батлалт", "78%")
    with col3:
        st.metric("Fraud илрүүлэлт", "0.8%")
    with col4:
        st.metric("Системийн нарийвчлал", "94%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Сегментээр зээлийн батлалт")
        
        segments = ['Шинэ', 'Стандарт', 'Premium', 'VIP', 'Эрсдэлтэй', 'Өндөр орлого']
        approval_rates = [0.45, 0.68, 0.85, 0.92, 0.23, 0.78]
        
        fig = px.bar(
            x=segments, 
            y=approval_rates,
            title="Харилцагчийн сегментээр зээлийн батлалтын хувь",
            labels={'x': 'Сегмент', 'y': 'Батлалтын хувь'},
            color=approval_rates,
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Fraud илрүүлэлтийн динамик")
        
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='M')
        fraud_counts = np.random.poisson(20, len(dates))
        
        fig = px.line(
            x=dates, 
            y=fraud_counts,
            title="Эрсдэлийн тохиолдлын тоо сар бүр",
            labels={'x': 'Сар', 'y': 'Fraud тохиолдлын тоо'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Моделийн статус")
    
    performance_data = {
        'Модель': ['Зээлийн эрсдэл', 'Fraud илрүүлэх', 'Харилцагчийн сегмент', 'Q&A систем'],
        'Статус': [' Идэвхтэй', ' Идэвхтэй', ' Идэвхтэй', ' Идэвхтэй'],
        'Хариу өгөх хугацаа (мс)': [120, 80, 60, 200],
        'Хэрэглээ': ['Өндөр', 'Дунд', 'Бага', 'Дунд']
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    st.markdown("####  Системийн хэвийн байдал")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU ашиглалт", "45%")
    with col2:
        st.metric("Memory ашиглалт", "62%")
    with col3:
        st.metric("API хариу өгөх хугацаа", "150ms")

def main():
    models = load_banking_models()
    if models is None:
        st.stop()
    
    st.markdown('<h1 class="main-header">Банкны AI Систем</h1>', unsafe_allow_html=True)
    
    st.sidebar.title(" Цэс")
    page = st.sidebar.selectbox(
        "Хуудас сонгоно уу:",
        [" Нүүр хуудас", " Зээлийн шинжилгээ", " Харилцагчийн сегмент", 
         " Fraud илрүүлэх", " Асуулт хариулт", " Дэшборд"]
    )
    
    if page == " Нүүр хуудас":
        show_home_page()
    elif page == " Зээлийн шинжилгээ":
        show_loan_analysis(models)
    elif page == " Харилцагчийн сегмент":
        show_customer_segmentation(models)
    elif page == " Fraud илрүүлэх":
        show_fraud_detection(models)
    elif page == " Асуулт хариулт":
        show_qa_system(models)
    elif page == " Дэшборд":
        show_dashboard(models)

if __name__ == "__main__":
    main()