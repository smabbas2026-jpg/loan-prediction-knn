import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class RobustLoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_weight=4.0):
        self.credit_weight = credit_weight

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
        
        # Loan Amount Risk (floored at 60k so small loans are not penalized)
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # Standardized benchmark DTI (360m amortization)
        benchmark_emi = (loan_amt * 1000.0) / 360.0
        dti_pct = (benchmark_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # High-weight Credit History: cred is 0.0 or 1.0
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Clean'] = cred

        # Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # Raw domain factors:
        # Term factor: positive for short terms (e.g. +0.97 for 12m), 0 for 360m, negative for 480m (-0.33)
        X['Term_Factor'] = (360.0 - loan_term) / 360.0

        # Education factor: 1.0 for Graduate, 0.0 for Not Graduate
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Factor'] = edu

        return X

class DomainMetricKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=15, alpha_term=0.10, beta_edu=0.07, credit_dist_penalty=5.0):
        self.n_neighbors = n_neighbors
        self.alpha_term = alpha_term
        self.beta_edu = beta_edu
        self.credit_dist_penalty = credit_dist_penalty

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.approved_mask_ = (self.y_train_ == 1)
        self.rejected_mask_ = (self.y_train_ == 0)
        return self

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_queries = X.shape[0]

        # X column order:
        # 0: Credit_History_Clean
        # 1: TotalIncome_Log (scaled)
        # 2: LoanAmount_Risk (scaled)
        # 3: Debt_Burden (scaled)
        # 4: Dependents_Num (scaled)
        # 5: Term_Factor
        # 6: Education_Factor
        # 7+: Categorical one-hot features (Gender, Married, Self_Employed, Property_Area)
        
        # Credit history column (index 0)
        q_cred = X[:, 0]
        train_cred = self.X_train_[:, 0]

        # Features used for general Euclidean distance (indices 1, 2, 3, 4 and 7+)
        gen_feat_idx = [1, 2, 3, 4] + list(range(7, X.shape[1]))
        X_gen = X[:, gen_feat_idx]
        train_gen = self.X_train_[:, gen_feat_idx]

        # Standard Euclidean distance on general financial features
        base_dists = np.linalg.norm(X_gen[:, np.newaxis, :] - train_gen[np.newaxis, :, :], axis=2)

        # Credit history distance penalty:
        # If credit history differs (0 vs 1), huge penalty ensuring credit history boundaries are respected
        cred_diff = np.abs(q_cred[:, np.newaxis] - train_cred[np.newaxis, :])
        base_dists += cred_diff * self.credit_dist_penalty

        # Domain conditioning:
        # Shorter term and Graduate status scale distance to approved vs rejected
        term_factor = X[:, 5]
        edu_factor = X[:, 6]
        benefit = self.alpha_term * term_factor + self.beta_edu * (edu_factor - 0.5)
        benefit = np.clip(benefit, -0.35, 0.35)

        for q in range(n_queries):
            b = benefit[q]
            # Approved points move closer by (1 - b)
            # Rejected points move farther by (1 + b)
            base_dists[q, self.approved_mask_] *= (1.0 - b)
            base_dists[q, self.rejected_mask_] *= (1.0 + b)

        top_k_idx = np.argsort(base_dists, axis=1)[:, :n_neighbors]
        top_k_dist = np.take_along_axis(base_dists, top_k_idx, axis=1)

        if return_distance:
            return top_k_dist, top_k_idx
        return top_k_idx

    def predict_proba(self, X):
        _, top_k_idx = self.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=True)
        neighbor_y = self.y_train_[top_k_idx]
        prob_1 = np.mean(neighbor_y == 1, axis=1)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('cred', 'passthrough', ['Credit_History_Clean']),
        ('scaled_num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Dependents_Num']),
        ('domain_factors', 'passthrough', ['Term_Factor', 'Education_Factor']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
        ]), ['Gender', 'Married', 'Self_Employed', 'Property_Area'])
    ]
)

pipe = Pipeline([
    ('fe', RobustLoanFeatureEngineer()),
    ('pre', preprocessor),
    ('clf', DomainMetricKNN(n_neighbors=15, alpha_term=0.10, beta_edu=0.08, credit_dist_penalty=5.0))
])

pipe.fit(X_train, y_train)

# Evaluate on test set
preds = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, preds)
print(f"=== TEST SET METRICS ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Verify applicants
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
    print(f"Terms sweep:")
    for t, p in zip(terms, t_probs):
        print(f"  {t:3d}m -> {p:.1%}")
