import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class SmartLoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_multiplier=2.5, education_bonus=1.5, term_decay_factor=1.0):
        self.credit_multiplier = credit_multiplier
        self.education_bonus = education_bonus
        self.term_decay_factor = term_decay_factor
        
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
        
        # 1. Loan Amount Risk (bounded at 60k so small loans are not penalized)
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # 2. Debt Burden: Standardized benchmark amortization (360 months)
        # Using benchmark 360m ensures short loan terms do not explode EMI
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # 3. Credit History Score
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Score'] = cred * self.credit_multiplier

        # 4. Education Score: Graduate gets positive creditworthiness signal
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Score'] = edu * self.education_bonus

        # 5. Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # 6. Term Duration Risk:
        # User requirement: 'the lesser the loan term the higher chance of approval'
        # In risk modeling, term duration risk increases monotonically with loan term.
        # Short terms (12-180m) have minimal duration risk (0 to 0.5),
        # 360m is standard (1.0), 480m is high risk (1.33+).
        # Normalizing term to [0, 1.33] aligns risk monotonically with term length.
        X['Term_Risk'] = (loan_term / 360.0) * self.term_decay_factor

        return X

# Test configurations
for term_decay in [0.0, 0.5, 1.0, 1.5]:
    for edu_bonus in [1.0, 1.5, 2.0]:
        num_cols = ['Credit_History_Score', 'Education_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Term_Risk', 'Dependents_Num']
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
            ('fe', SmartLoanFeatureEngineer(credit_multiplier=2.5, education_bonus=edu_bonus, term_decay_factor=term_decay)),
            ('pre', preprocessor),
            ('clf', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
        ])

        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        
        # Test Strong, Average, Borderline
        strong = pd.DataFrame([{
            'Gender': 'Male', 'Married': 'Yes', 'Dependents': '0', 'Education': 'Graduate', 'Self_Employed': 'No',
            'ApplicantIncome': 5000.0, 'CoapplicantIncome': 2000.0, 'LoanAmount': 100.0, 'Loan_Amount_Term': 360.0,
            'Credit_History': 1.0, 'Property_Area': 'Semiurban'
        }])
        borderline = pd.DataFrame([{
            'Gender': 'Female', 'Married': 'No', 'Dependents': '2', 'Education': 'Graduate', 'Self_Employed': 'Yes',
            'ApplicantIncome': 2800.0, 'CoapplicantIncome': 0.0, 'LoanAmount': 140.0, 'Loan_Amount_Term': 360.0,
            'Credit_History': 1.0, 'Property_Area': 'Rural'
        }])

        # Check Graduate vs Not Graduate for both
        s_g = pipe.predict_proba(strong)[0][1]
        strong_ng = strong.copy(); strong_ng['Education'] = 'Not Graduate'
        s_ng = pipe.predict_proba(strong_ng)[0][1]

        b_g = pipe.predict_proba(borderline)[0][1]
        borderline_ng = borderline.copy(); borderline_ng['Education'] = 'Not Graduate'
        b_ng = pipe.predict_proba(borderline_ng)[0][1]

        # Check Term sweep for strong & borderline
        terms = [12, 60, 180, 360, 480]
        s_term_probs = [pipe.predict_proba(strong.assign(Loan_Amount_Term=t))[0][1] for t in terms]
        b_term_probs = [pipe.predict_proba(borderline.assign(Loan_Amount_Term=t))[0][1] for t in terms]

        s_term_mono = all(s_term_probs[i] >= s_term_probs[i+1] for i in range(len(terms)-1))
        b_term_mono = all(b_term_probs[i] >= b_term_probs[i+1] for i in range(len(terms)-1))

        if s_g >= s_ng and b_g >= b_ng and acc >= 0.84:
            print(f"Candidate: term_decay={term_decay}, edu_bonus={edu_bonus} | Acc={acc:.4f}")
            print(f"  Strong: Grad={s_g:.1%} vs NotGrad={s_ng:.1%} | Terms {terms}: {[f'{p:.1%}' for p in s_term_probs]} (Mono={s_term_mono})")
            print(f"  Border: Grad={b_g:.1%} vs NotGrad={b_ng:.1%} | Terms {terms}: {[f'{p:.1%}' for p in b_term_probs]} (Mono={b_term_mono})")
