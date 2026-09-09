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

# Let's test:
# In lending, an applicant has:
# 1. Total Household Income (log)
# 2. Loan Amount Risk: max(LoanAmount, 60.0)
# 3. Benchmark Debt Burden: (LoanAmount * 1000 / 360) / TotalIncome * 100
# 4. Credit History Score:
#    In credit risk, overall credit standing is composed of:
#    - Base Credit History (dominant: 0 or 1)
#    - Graduate bonus (+ credit standing for graduates)
#    - Term duration bonus: shorter terms (e.g. 12m, 60m) have higher credit standing than 360m;
#      terms > 360m (like 480m) have penalized credit standing!
#
# Let's test if Credit_Profile_Score = Credit_History + Edu_Adj + Term_Adj
# Or if they are separate aligned features.

class CreditProfileFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, edu_weight=0.3, term_weight=0.2):
        self.edu_weight = edu_weight
        self.term_weight = term_weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Incomes
        app_inc = pd.to_numeric(X.get('ApplicantIncome', 0), errors='coerce').fillna(3500)
        coapp_inc = pd.to_numeric(X.get('CoapplicantIncome', 0), errors='coerce').fillna(0)
        total_income = np.maximum(app_inc + coapp_inc, 500.0)
        X['TotalIncome_Log'] = np.log(total_income)

        # Loan Amount
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

        # Term signal:
        # Shorter term -> Higher credit stability score
        # 12m -> (360 - 12)/360 = +0.97
        # 60m -> (360 - 60)/360 = +0.83
        # 180m -> (360 - 180)/360 = +0.50
        # 360m -> 0.0
        # 480m -> (360 - 480)/360 = -0.33
        term_factor = (360.0 - loan_term) / 360.0

        # Composite Credit Score:
        # High credit history is primary (weight 3.0)
        # Graduate provides positive boost (+ edu_weight)
        # Shorter term provides positive duration boost (+ term_weight)
        X['Credit_Score'] = cred * 3.0 + edu * self.edu_weight + term_factor * self.term_weight

        # Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        return X

# Test this pipeline
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

pipe = Pipeline([
    ('fe', CreditProfileFeatureEngineer(edu_weight=0.3, term_weight=0.2)),
    ('pre', preprocessor),
    ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
])

pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

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

for name, app in [("Strong Applicant", strong), ("Borderline Applicant", borderline), ("Weak Applicant", weak)]:
    print(f"\n=== {name} ===")
    g_prob = pipe.predict_proba(app.assign(Education='Graduate'))[0][1]
    ng_prob = pipe.predict_proba(app.assign(Education='Not Graduate'))[0][1]
    print(f"Education: Graduate = {g_prob:.1%} | Not Graduate = {ng_prob:.1%} (Advantage: +{(g_prob - ng_prob):.1%})")
    
    terms = [12, 36, 60, 120, 180, 240, 300, 360, 480]
    t_probs = [pipe.predict_proba(app.assign(Loan_Amount_Term=t))[0][1] for t in terms]
    print("Term sweep:")
    for t, p in zip(terms, t_probs):
        print(f"  {t:3d}m -> {p:.1%}")
