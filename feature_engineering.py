import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-aware feature transformer for credit risk and loan prediction.
    Features:
    - Credit_History_Score: Weighted credit history preserving high predictive power.
    - TotalIncome_Log: Log-transformed household income (Applicant + Coapplicant).
    - LoanAmount_Risk: Base-bounded loan amount ensuring small loans (<$60k) are not penalized as outliers.
    - Debt_Burden: Estimated Monthly Installment (EMI) to Monthly Income percentage.
    - Dependents_Num: Numeric integer representation of dependents.
    """
    def __init__(self, credit_weight=2.5):
        self.credit_weight = credit_weight

    def fit(self, X, y=None):
        X_df = X.copy()
        app_inc = pd.to_numeric(X_df.get('ApplicantIncome', 0), errors='coerce')
        coapp_inc = pd.to_numeric(X_df.get('CoapplicantIncome', 0), errors='coerce')
        self.med_income_ = (app_inc.fillna(0) + coapp_inc.fillna(0)).median()
        self.med_loan_ = pd.to_numeric(X_df.get('LoanAmount', 0), errors='coerce').median()
        self.med_term_ = pd.to_numeric(X_df.get('Loan_Amount_Term', 360), errors='coerce').median()
        return self

    def transform(self, X):
        X = X.copy()
        
        # Clean Dependents
        dep = X.get('Dependents', '0').astype(str).str.replace('+', '', regex=False)
        X['Dependents_Num'] = pd.to_numeric(dep, errors='coerce').fillna(0)

        # Incomes
        app_inc = pd.to_numeric(X.get('ApplicantIncome', 0), errors='coerce').fillna(getattr(self, 'med_income_', 5000) * 0.7)
        coapp_inc = pd.to_numeric(X.get('CoapplicantIncome', 0), errors='coerce').fillna(0)
        total_income = np.maximum(app_inc + coapp_inc, 500.0)
        X['TotalIncome'] = total_income
        X['TotalIncome_Log'] = np.log(total_income)

        # Loan Amount & Term
        med_loan = getattr(self, 'med_loan_', 128.0)
        med_term = getattr(self, 'med_term_', 360.0)
        loan_amt = pd.to_numeric(X.get('LoanAmount', 0), errors='coerce').fillna(med_loan)
        loan_term = pd.to_numeric(X.get('Loan_Amount_Term', 360), errors='coerce').fillna(med_term)
        loan_term = np.maximum(loan_term, 12.0)
        
        # LoanAmount_Risk: In credit underwriting, low loan amounts (e.g. 10k, 25k, 50k)
        # are strictly low-risk. Floor at 60k prevents extreme negative z-scores.
        X['LoanAmount_Risk'] = np.maximum(loan_amt, 60.0)

        # Debt Burden: Monthly EMI as percentage of monthly income
        monthly_emi = (loan_amt * 1000.0) / loan_term
        X['Monthly_EMI'] = monthly_emi
        dti_pct = (monthly_emi / total_income) * 100.0
        X['Debt_Burden'] = np.maximum(dti_pct, 10.0)

        # Credit History: Most critical credit signal, weighted appropriately
        cred = pd.to_numeric(X.get('Credit_History', 1.0), errors='coerce').fillna(1.0)
        X['Credit_History_Score'] = cred * self.credit_weight

        return X
