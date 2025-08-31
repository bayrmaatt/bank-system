# Bank AI System

Төслийн зорилго нь Монголын банкуудад зориулсан **харилцагчийн сегментчилэл, зээлийн баталгаажуулалт, fraud detection**, мөн **асуулт хариулт (Q&A) систем**-ийг нэвтрүүлэх юм. Бүх системийг **Streamlit** ашиглан интерактив dashboard болгосон.

## To Run:


## Үндсэн модульууд

### 1. Харилцагчийн сегментчилэл
- Харилцагчдын **нас, орлого, ажлын туршлага, зээлийн тоо, кредит оноо** зэрэг үзүүлэлтүүдээр сегментчилдэг.
- Сегментүүдийн жишээ:
  - VIP харилцагч: Орлого ≥ 15M₮, кредит оноо ≥ 750, банкны харилцаа ≥ 5 жил
  - Premium: Орлого ≥ 5M₮, кредит оноо ≥ 700
  - Стандарт:  Бусад тохиолдол
  - Эрсдэлтэй: Кредит оноо < 500, 3+ зээлтэй, эсвэл зарлага > 80% орлого
- Сегментчилэл нь маркетинг, risk analysis, resource planning-д ашиглагдана.

### 2. Зээлийн баталгаажуулалт (Loan Approval)
- **Random Forest** болон **ANN (Keras Sequential)** загвар ашигласан.
- Онцлог:
  - Орлого, зээлийн хэмжээ, кредит оноо, барьцаа, ажлын туршлага, боловсрол зэргийг feature болгон ашигласан.
  - ANN model:
    - Accuracy: 98%
    - Hidden layers: 256 → 128 → 64 → 1 neuron (sigmoid)
  - Random Forest model:
    - Accuracy: 85%
    - Top features: `credit_score`, `requested_amount`, `monthly_income`

### 3. Fraud Detection
- **Random Forest + Gradient Boosting Ensemble** ашиглан transaction fraud илрүүлдэг.
- Feature engineering:  
  - Amount log, z-score  
  - Night/Weekend transaction  
  - International/Online transaction  
- Accuracy: ~98%

### 4. Banking Q&A System
- Монгол хэл дээрх **loan products болон frequently asked questions**-ийг vectorization (TF-IDF) ашиглан similarity-based хариулт олдог.
- Features:
  - Product keywords, description, loan_type
  - Cosine similarity ашиглан асуулт хариулт гаргах
- Streamlit-д интерактив query system-оор ашиглах боломжтой.

### 5. Data Processing & Feature Engineering
- Харилцагчийн data болон transaction data-д **Label Encoding, StandardScaler, log/z-score** боловсруулалт хийсэн.
- Financial ratios:
  - `debt_to_income`, `expense_to_income`, `collateral_coverage`
  - Зээлийн баталгаажуулалт болон risk evaluation-д ашиглагдсан.

### 6. Streamlit Dashboard
- Харилцагчийн сегментчилэл, зээлийн баталгаажуулалт, fraud detection, Q&A системийг нэг dashboard-д нэгтгэсэн.
- Интерактив charts, metrics, real-time prediction-уудыг харуулах боломжтой.

## Technologies
- Python, Pandas, NumPy, Scikit-learn
- TensorFlow/Keras (ANN)
- Streamlit
- Joblib (model & encoder serialization)

## Usage
```bash
git clone <repository_url>
cd bank_ai_system
streamlit run app.py

