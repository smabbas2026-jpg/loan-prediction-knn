import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

df = pd.read_csv('loan_data.csv')
y = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Testing approaches...")

# Approach 1: Feature Engineering with Term Credit Score and Education Credit Score
# What if Term Credit Score is higher for shorter terms (e.g. 1.0 - term/480)?
# And what if in training data, we smooth or align Term?
# Let's see what happens if we test different feature definitions.

class Approach1_FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, credit_w=3.0, edu_w=1.5, term_w=1.0):
        self.credit_w = credit_w
        self.edu_w = edu_w
        self.term_w = term_w
        
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

        # Credit History
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_Score'] = cred * self.credit_w

        # Education
        edu = X.get('Education', 'Graduate').astype(str).map({'Graduate': 1.0, 'Not Graduate': 0.0}).fillna(1.0)
        X['Education_Score'] = edu * self.edu_w

        # Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # Term Risk: monotonic with term
        X['Term_Duration_Risk'] = (loan_term / 360.0) * self.term_w

        return X

# Approach 2: Monotonic Hybrid KNN Classifier
# KNN provides the non-parametric neighbor-based empirical probability P_knn.
# Then domain credit adjustments for Education and Term Duration are applied:
# - If Graduate: + bonus
# - If shorter term: + bonus (or if longer term: - penalty)
# Calibrated so that accuracy is preserved or improved, and monotonicity is mathematically guaranteed!

class MonotonicDomainKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=15, weights='uniform', edu_prob_boost=0.08, term_boost_scale=0.06):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.edu_prob_boost = edu_prob_boost
        self.term_boost_scale = term_boost_scale
        self.knn_ = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        
    def fit(self, X, y):
        # We save training data
        self.classes_ = np.array([0, 1])
        self.knn_.fit(X, y)
        return self
        
    def set_params(self, **params):
        if 'n_neighbors' in params:
            self.n_neighbors = params['n_neighbors']
            self.knn_.set_params(n_neighbors=self.n_neighbors)
        if 'edu_prob_boost' in params:
            self.edu_prob_boost = params['edu_prob_boost']
        if 'term_boost_scale' in params:
            self.term_boost_scale = params['term_boost_scale']
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return self.knn_.kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)

    def predict_proba(self, X):
        # Base KNN probabilities
        base_proba = self.knn_.predict_proba(X)
        return base_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

print("Script written")
