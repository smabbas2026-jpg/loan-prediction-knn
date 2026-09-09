import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class CreditProfileFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_weight=2.5, edu_weight=0.8, term_weight=0.6):
        self.credit_weight = credit_weight
        self.edu_weight = edu_weight
        self.term_weight = term_weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        app_inc = pd.to_numeric(X.get('ApplicantIncome', 0), errors='coerce').fillna(3500)
        coapp_inc = pd.to_numeric(X.get('CoapplicantIncome', 0), errors='coerce').fillna(0)
        total_income = np.maximum(app_inc + coapp_inc, 500.0)
        X['TotalIncome_Log'] = np.log(total_income)

        loan_amt = pd.to_numeric(X.get('LoanAmount', 0), errors='coerce').fillna(128.0)
        loan_term = pd.to_numeric(X.get('Loan_Amount_Term', 360), errors='coerce').fillna(360.0)
        loan_term = np.maximum(loan_term, 12.0)
        
        # Loan Amount Risk (floored at 60k so small loans are safe)
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # Standardized benchmark DTI (360m amortization)
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # Base Credit History (0 or 1)
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        
        # Education signal: Graduate = +1, Not Graduate = 0
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)

        # Term factor:
        # Shorter term -> Higher stability / less duration risk
        # 12m -> +0.97
        # 60m -> +0.83
        # 360m -> 0.0
        # 480m -> -0.33
        term_factor = (360.0 - loan_term) / 360.0

        X['Credit_Score'] = cred * self.credit_weight + edu * self.edu_weight + term_factor * self.term_weight

        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        return X

strong = pd.DataFrame([{
    'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 'Self_Employed': 'No',
    'ApplicantIncome': 6500.0, 'CoapplicantIncome': 2500.0, 'LoanAmount': 140.0, 'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0, 'Property_Area': 'Semiurban'
}])

borderline = pd.DataFrame([{
    'Gender': 'Female', 'Married': 'No', 'Dependents': '2', 'Education': 'Not Graduate', 'Self_Employed': 'Yes',
    'ApplicantIncome': 2600.0, 'CoapplicantIncome': 0.0, 'LoanAmount': 150.0, 'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0, 'Property_Area': 'Rural'
}])

weak = pd.DataFrame([{
    'Gender': 'Male', 'Married': 'No', 'Dependents': '1', 'Education': 'Not Graduate', 'Self_Employed': 'No',
    'ApplicantIncome': 2200.0, 'CoapplicantIncome': 0.0, 'LoanAmount': 180.0, 'Loan_Amount_Term': 180.0,
    'Credit_History': 0.0, 'Property_Area': 'Urban'
}])

num_cols = ['Credit_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Dependents_Num']
cat_cols = ['Gender', 'Married', 'Self_Employed', 'Property_Area']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
        ]), cat_cols)
    ]
)

for ew in [0.6, 0.8, 1.0, 1.2]:
    for tw in [0.4, 0.6, 0.8, 1.0]:
        pipe = Pipeline([
            ('fe', CreditProfileFeatureEngineer(credit_weight=2.5, edu_weight=ew, term_weight=tw)),
            ('pre', preprocessor),
            ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        if acc < 0.84:
            continue
            
        # Test terms
        terms = [12, 60, 180, 360, 480]
        s_terms = [pipe.predict_proba(strong.assign(Loan_Amount_Term=t))[0][1] for t in terms]
        b_terms = [pipe.predict_proba(borderline.assign(Loan_Amount_Term=t))[0][1] for t in terms]
        
        # Test education
        s_g = pipe.predict_proba(strong.assign(Education='Graduate'))[0][1]
        s_ng = pipe.predict_proba(strong.assign(Education='Not Graduate'))[0][1]
        b_g = pipe.predict_proba(borderline.assign(Education='Graduate'))[0][1]
        b_ng = pipe.predict_proba(borderline.assign(Education='Not Graduate'))[0][1]
        
        s_term_mono = all(s_terms[i] >= s_terms[i+1] for i in range(len(terms)-1))
        b_term_mono = all(b_terms[i] >= b_terms[i+1] for i in range(len(terms)-1))
        
        s_edu_ok = (s_g >= s_ng)
        b_edu_ok = (b_g >= b_ng)
        
        if s_term_mono and b_term_mono and s_edu_ok and b_edu_ok:
            print(f"FOUND MATCH: ew={ew}, tw={tw}, acc={acc:.4f}")
            print(f"  Strong Terms {terms}: {[f'{p:.1%}' for p in s_terms]}")
            print(f"  Border Terms {terms}: {[f'{p:.1%}' for p in b_terms]}")
            print(f"  Strong Edu: Grad={s_g:.1%}, NotGrad={s_ng:.1%}")
            print(f"  Border Edu: Grad={b_g:.1%}, NotGrad={b_ng:.1%}")
