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

class CreditRiskPipelineEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_weight=2.5, education_weight=2.0, term_weight=1.5):
        self.credit_weight = credit_weight
        self.education_weight = education_weight
        self.term_weight = term_weight
        
    def fit(self, X, y=None):
        X_df = X.copy()
        app_inc = pd.to_numeric(X_df.get('ApplicantIncome', 0), errors='coerce')
        coapp_inc = pd.to_numeric(X_df.get('CoapplicantIncome', 0), errors='coerce')
        self.med_income_ = (app_inc.fillna(0) + coapp_inc.fillna(0)).median()
        self.med_loan_ = pd.to_numeric(X_df.get('LoanAmount', 0), errors='coerce').median()
        return self
        
    def transform(self, X):
        X = X.copy()
        app_inc = pd.to_numeric(X.get('ApplicantIncome', 0), errors='coerce').fillna(getattr(self, 'med_income_', 5000) * 0.7)
        coapp_inc = pd.to_numeric(X.get('CoapplicantIncome', 0), errors='coerce').fillna(0)
        total_income = np.maximum(app_inc + coapp_inc, 500.0)
        X['TotalIncome'] = total_income
        X['TotalIncome_Log'] = np.log(total_income)

        loan_amt = pd.to_numeric(X.get('LoanAmount', 0), errors='coerce').fillna(getattr(self, 'med_loan_', 128.0))
        loan_term = pd.to_numeric(X.get('Loan_Amount_Term', 360), errors='coerce').fillna(360.0)
        loan_term = np.maximum(loan_term, 12.0)
        
        # 1. Loan Amount Risk (floored at 60k so small loans are not penalized)
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # 2. Debt Burden based on benchmark amortization
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # 3. Credit History (heavily weighted signal)
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Score'] = cred * self.credit_weight

        # 4. Education Score: Graduate = 1.0, Not Graduate = 0.0
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Score'] = edu * self.education_weight

        # 5. Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # 6. Term Duration Risk:
        # Shorter terms mean less duration risk.
        # Excess term (>360) is high risk.
        # Terms <= 360 have zero excess term risk, plus a duration tier.
        # Let's formulate term risk so that 480m is penalized, 360m is baseline, and <360m is rewarded.
        # In standardized space, if Term_Duration is modeled as:
        # term / 360 capped, or excess term:
        excess_term = np.maximum(loan_term - 360.0, 0.0) / 60.0 # 0 for <=360, 2.0 for 480
        X['Excess_Term_Risk'] = excess_term * self.term_weight

        # Also, duration tier:
        # short terms (<=180) get a bonus (negative risk), standard terms (240-360) are 0, >360 are positive
        term_tier = np.where(loan_term <= 120, -1.0, np.where(loan_term <= 240, -0.5, np.where(loan_term <= 360, 0.0, 1.5)))
        X['Term_Tier'] = term_tier * self.term_weight

        return X

num_cols = ['Credit_History_Score', 'Education_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Excess_Term_Risk', 'Term_Tier', 'Dependents_Num']
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
    ('fe', CreditRiskPipelineEngineer(credit_weight=2.5, education_weight=2.0, term_weight=1.5)),
    ('pre', preprocessor),
    ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
])

pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

applicant = pd.DataFrame([{
    'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 'Self_Employed': 'No',
    'ApplicantIncome': 5000.0, 'CoapplicantIncome': 1500.0, 'LoanAmount': 130.0, 'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0, 'Property_Area': 'Semiurban'
}])

print("\n=== TERM MONOTONICITY TEST ===")
for term in [12, 36, 60, 120, 180, 240, 300, 360, 480]:
    row = applicant.copy()
    row['Loan_Amount_Term'] = float(term)
    prob = pipe.predict_proba(row)[0][1]
    print(f"  Term={term:3d}m ({term/12:4.1f}y) -> Approval Prob: {prob*100:.1f}%")

print("\n=== EDUCATION TEST ===")
grad = applicant.copy()
grad['Education'] = 'Graduate'
ng = applicant.copy()
ng['Education'] = 'Not Graduate'
pg = pipe.predict_proba(grad)[0][1]
png = pipe.predict_proba(ng)[0][1]
print(f"  Graduate:     {pg*100:.1f}%")
print(f"  Not Graduate: {png*100:.1f}%")
print(f"  Advantage:    +{(pg - png)*100:.1f}%")
