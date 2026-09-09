import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class SmartLoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_weight=2.5):
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

        # Base Credit History (0 or 1)
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Score'] = cred * self.credit_weight

        # Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # Raw domain factors passed through for metric conditioning:
        # 1. Term Duration Factor: positive for short terms, negative for >360m
        X['Term_Factor'] = (360.0 - loan_term) / 360.0
        # 2. Education Graduate Factor: 1.0 if Graduate else 0.0
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Factor'] = edu

        return X

class DomainMetricKNN(BaseEstimator, ClassifierMixin):
    """
    Cost-Sensitive Domain-Conditioned K-Nearest Neighbors Classifier.
    - Shorter loan terms reduce duration risk -> pulls approved neighbors closer.
    - Graduate degree enhances stability -> pulls approved neighbors closer.
    Guarantees:
    - Lesser loan term -> higher (or equal) approval probability.
    - Graduate -> higher (or equal) approval probability than Not Graduate.
    """
    def __init__(self, n_neighbors=15, weights='uniform', alpha_term=0.12, beta_edu=0.08):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.alpha_term = alpha_term
        self.beta_edu = beta_edu

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

    def _extract_benefit(self, X):
        # Column indices:
        # num_cols: Credit_History_Score(0), TotalIncome_Log(1), LoanAmount_Risk(2), Debt_Burden(3), Dependents_Num(4), Term_Factor(5), Education_Factor(6)
        term_factor = X[:, 5]
        edu_factor = X[:, 6]
        # Benefit increases for shorter terms (term_factor > 0) and for graduates (edu_factor == 1)
        benefit = self.alpha_term * term_factor + self.beta_edu * (edu_factor - 0.5)
        return np.clip(benefit, -0.35, 0.35)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_queries = X.shape[0]

        # In distance calculation, we use features 0 through 4 (standard credit features)
        # plus categorical features 7+, excluding the raw conditioning factors 5 and 6
        # from Euclidean coordinate distance
        feature_mask = np.ones(X.shape[1], dtype=bool)
        feature_mask[5] = False # Term_Factor used as metric modifier
        feature_mask[6] = False # Education_Factor used as metric modifier

        X_feat = X[:, feature_mask]
        train_feat = self.X_train_[:, feature_mask]

        # Base Euclidean distances to all training points
        dists = np.linalg.norm(X_feat[:, np.newaxis, :] - train_feat[np.newaxis, :, :], axis=2)

        # Domain modifier
        benefit = self._extract_benefit(X)

        for q in range(n_queries):
            b = benefit[q]
            # Approved points move closer by (1 - b)
            # Rejected points move farther by (1 + b)
            dists[q, self.approved_mask_] *= (1.0 - b)
            dists[q, self.rejected_mask_] *= (1.0 + b)

        # Sort and pick top k
        top_k_idx = np.argsort(dists, axis=1)[:, :n_neighbors]
        top_k_dist = np.take_along_axis(dists, top_k_idx, axis=1)

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

# Build pipeline
num_cols = ['Credit_History_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Dependents_Num', 'Term_Factor', 'Education_Factor']
cat_cols = ['Gender', 'Married', 'Self_Employed', 'Property_Area']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Credit_History_Score', 'TotalIncome_Log', 'LoanAmount_Risk', 'Debt_Burden', 'Dependents_Num']),
        ('passthrough_factors', 'passthrough', ['Term_Factor', 'Education_Factor']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
        ]), cat_cols)
    ]
)

pipe = Pipeline([
    ('fe', SmartLoanFeatureEngineer(credit_weight=2.5)),
    ('pre', preprocessor),
    ('clf', DomainMetricKNN(n_neighbors=15, alpha_term=0.15, beta_edu=0.10))
])

pipe.fit(X_train, y_train)

# Evaluate on test set
preds = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, preds)
print(f"=== TEST EVALUATION ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print(f"F1-Score: {f1:.4f}")

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
    print(f"Education: Graduate={g_prob:.1%} vs Not Graduate={ng_prob:.1%} (Advantage: +{(g_prob - ng_prob):.1%})")
    
    terms = [12, 36, 60, 120, 180, 240, 300, 360, 480]
    t_probs = [pipe.predict_proba(app.assign(Loan_Amount_Term=t))[0][1] for t in terms]
    print(f"Terms sweep:")
    for t, p in zip(terms, t_probs):
        print(f"  {t:3d}m -> {p:.1%}")
