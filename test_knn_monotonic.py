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

# Let's test how feature engineering should represent:
# 1. Loan_Amount_Term:
#    - To avoid monthly_emi explosion for 12m: benchmark 360m amortization
#    - For term itself: how should term be encoded?
#      If term is shorter, what feature represents higher chance of approval?
#      Let's test:
#      - Term_Credit_Factor: (360.0 - term) / 360.0 (positive for short terms, negative for 480m)
#      - Or Excess_Term_Risk: max(term - 360, 0)
#      - Or Short_Term_Bonus: max(360 - term, 0)
#      - Or what if Term affects the Creditworthiness Score?
#
# 2. Education:
#    - How should Education be encoded so Graduate ALWAYS has higher or equal approval?
#
# Let's test combinations!

print("Testing designs...")
