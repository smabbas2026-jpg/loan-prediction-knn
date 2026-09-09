import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class FeatureTransformerWithWeights(BaseEstimator, TransformerMixin):
    def __init__(self, credit_weight=4.0, education_weight=1.5, term_weight=1.0):
        self.credit_weight = credit_weight
        self.education_weight = education_weight
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

        # Loan Amount & Term
        loan_amt = pd.to_numeric(X.get('LoanAmount', 0), errors='coerce').fillna(128.0)
        loan_term = pd.to_numeric(X.get('Loan_Amount_Term', 360), errors='coerce').fillna(360.0)
        loan_term = np.maximum(loan_term, 12.0)
        
        # LoanAmount_Risk (bounded at 60k so small loans are not penalized)
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # Debt Burden: Standardized benchmark amortization (360m)
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # Raw credit and education for post-scaler weighting
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Clean'] = cred

        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Clean'] = edu

        # Term risk:
        # If term > 360: excess duration risk
        # If term < 360: how should it be represented?
        X['Excess_Term_Risk'] = np.maximum(loan_term - 360.0, 0.0) / 60.0
        # Short term advantage: shorter term gets positive score or lower risk
        # E.g. Term_Duration = loan_term / 360.0
        X['Term_Months'] = loan_term

        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        return X

class PostScalingWeighter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_weights=None):
        self.feature_weights = feature_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.feature_weights is not None:
            return X * np.array(self.feature_weights)
        return X

# Let's test with a small experiment
num_cols = ['Credit_History_Clean', 'Education_Clean', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Excess_Term_Risk', 'Dependents_Num']
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

# num_cols: 7 features
# cat_cols with drop='if_binary':
# Gender (1), Married (1), Self_Employed (1), Property_Area (3) = 6 features
# Total = 13 features
# Let's give:
# Credit_History: 3.5
# Education: 1.5
# TotalIncome_Log: 1.0
# LoanAmount_Risk: 1.0
# Debt_Burden: 1.0
# Excess_Term_Risk: 1.5
# Dependents_Num: 0.8
# Cat cols: 0.5 each
weights = [3.5, 1.5, 1.0, 1.0, 1.0, 1.5, 0.8] + [0.5] * 6

pipe = Pipeline([
    ('fe', FeatureTransformerWithWeights()),
    ('pre', preprocessor),
    ('weighter', PostScalingWeighter(weights)),
    ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
])

pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
print(f"Test Accuracy with PostScalingWeighter: {acc:.4f}")

# Test applicants
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
    print(f"\n--- {name} ---")
    g_prob = pipe.predict_proba(app.assign(Education='Graduate'))[0][1]
    ng_prob = pipe.predict_proba(app.assign(Education='Not Graduate'))[0][1]
    print(f"Education: Graduate = {g_prob:.1%} | Not Graduate = {ng_prob:.1%} (Advantage: +{(g_prob - ng_prob):.1%})")
    
    terms = [12, 36, 60, 120, 180, 240, 300, 360, 480]
    t_probs = [pipe.predict_proba(app.assign(Loan_Amount_Term=t))[0][1] for t in terms]
    print(f"Terms {terms}:")
    print(f"  {[f'{t}m: {p:.1%}' for t, p in zip(terms, t_probs)]}")
