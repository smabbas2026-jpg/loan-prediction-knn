import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Let's test how to ensure:
# 1. Shorter term -> Higher (or equal) chance of approval
# 2. Graduate -> Higher chance of approval than Not Graduate
# Let's test across several applicants (Strong, Average, Borderline)

class DomainMonotonicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_weight=2.5, education_weight=2.0, term_penalty_weight=1.5):
        self.credit_weight = credit_weight
        self.education_weight = education_weight
        self.term_penalty_weight = term_penalty_weight
        
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
        
        # 1. Loan Amount Risk
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # 2. Debt Burden based on benchmark amortization (360 months)
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # 3. Credit History Score (weighted heavily)
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Score'] = cred * self.credit_weight

        # 4. Education Score: Graduate = 1.0, Not Graduate = 0.0 (weighted heavily)
        # In underwriting, graduates have stronger creditworthiness
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Score'] = edu * self.education_weight

        # 5. Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # 6. Term Duration Risk:
        # In credit risk, terms > 360 (like 480m) represent excess duration risk.
        # Short terms (<=360m) have zero duration risk.
        # To make shorter term have higher (or equal) approval probability without creating
        # Euclidean isolation, we include Excess_Term_Risk for terms > 360.
        excess_term = np.maximum(loan_term - 360.0, 0.0) / 60.0
        X['Excess_Term_Risk'] = excess_term * self.term_penalty_weight

        return X

applicant = pd.DataFrame([{
    'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 'Self_Employed': 'No',
    'ApplicantIncome': 4500.0, 'CoapplicantIncome': 1500.0, 'LoanAmount': 130.0, 'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0, 'Property_Area': 'Semiurban'
}])

borderline = pd.DataFrame([{
    'Gender': 'Female', 'Married': 'No', 'Dependents': '2', 'Education': 'Not Graduate', 'Self_Employed': 'Yes',
    'ApplicantIncome': 2600.0, 'CoapplicantIncome': 0.0, 'LoanAmount': 150.0, 'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0, 'Property_Area': 'Rural'
}])

num_cols = ['Credit_History_Score', 'Education_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Excess_Term_Risk', 'Dependents_Num']
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

pipe = Pipeline([
    ('fe', DomainMonotonicFeatureEngineer(credit_weight=2.5, education_weight=2.0, term_penalty_weight=2.0)),
    ('pre', preprocessor),
    ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
])

pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

for name, app in [("Strong Applicant", applicant), ("Borderline Applicant", borderline)]:
    print(f"\n=== {name} ===")
    print("Term sweep:")
    for term in [12, 36, 60, 120, 180, 240, 300, 360, 480]:
        row = app.copy()
        row['Loan_Amount_Term'] = float(term)
        prob = pipe.predict_proba(row)[0][1]
        print(f"  Term={term:3d}m -> Approval Prob: {prob*100:.1f}%")
        
    g_row = app.copy()
    g_row['Education'] = 'Graduate'
    ng_row = app.copy()
    ng_row['Education'] = 'Not Graduate'
    pg = pipe.predict_proba(g_row)[0][1]
    png = pipe.predict_proba(ng_row)[0][1]
    print(f"Education: Graduate={pg*100:.1f}% | Not Graduate={png*100:.1f}% (Advantage: +{(pg-png)*100:.1f}%)")
