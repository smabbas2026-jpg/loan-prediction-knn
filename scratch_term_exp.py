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

class MonotonicTermEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, term_scale=1.0, credit_weight=2.5, education_weight=2.0):
        self.term_scale = term_scale
        self.credit_weight = credit_weight
        self.education_weight = education_weight
        
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
        
        # Loan Amount Risk (floored at 60k so small loans are not penalized)
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # Debt Burden (using 360 benchmark so short terms don't inflate EMI)
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # Credit History (heavily weighted)
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Score'] = cred * self.credit_weight

        # Education: Graduate = 1.0, Not Graduate = 0.0 (weighted)
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Score'] = edu * self.education_weight

        # Monotonic Term Risk:
        # We want: lesser loan term -> higher (or equal) chance of approval.
        # Long term (480) has highest risk, standard term (360) is normal,
        # and short terms (12-180) have lowest risk.
        # Using a smooth monotonic penalty:
        term_penalty = (loan_term / 120.0) * self.term_scale
        X['Term_Risk'] = term_penalty

        return X

applicant = pd.DataFrame([{
    'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 'Self_Employed': 'No',
    'ApplicantIncome': 4200.0, 'CoapplicantIncome': 1200.0, 'LoanAmount': 130.0, 'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0, 'Property_Area': 'Semiurban'
}])

for ts in [0.5, 0.8, 1.0, 1.2, 1.5]:
    num_cols = ['Credit_History_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Term_Risk', 'Education_Score', 'Dependents_Num']
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

    p = Pipeline([
        ('fe', MonotonicTermEngineer(term_scale=ts, credit_weight=2.5, education_weight=2.0)),
        ('pre', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
    ])
    p.fit(X_train, y_train)
    acc = p.score(X_test, y_test)
    
    print(f"\n--- Term Scale: {ts} (Test Acc: {acc:.4f}) ---")
    for term in [12, 36, 60, 120, 180, 240, 300, 360, 480]:
        row = applicant.copy()
        row['Loan_Amount_Term'] = float(term)
        prob = p.predict_proba(row)[0][1]
        print(f"  Term={term:3d}m -> Approval: {prob*100:.1f}%")
        
    grad = applicant.copy()
    grad['Education'] = 'Graduate'
    ng = applicant.copy()
    ng['Education'] = 'Not Graduate'
    print(f"  Graduate: {p.predict_proba(grad)[0][1]*100:.1f}% | Not Graduate: {p.predict_proba(ng)[0][1]*100:.1f}%")
