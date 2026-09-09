import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from feature_engineering import LoanFeatureEngineer


def load_and_preprocess_data(csv_path="loan_data.csv"):
    df = pd.read_csv(csv_path)
    
    # Drop Loan_ID if present as it is a unique identifier
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Binary Target mapping: Y -> 1, N -> 0
    target_col = "Loan_Status"
    y = df[target_col].map({"Y": 1, "N": 0}).astype(int)
    X = df.drop(columns=[target_col])

    return X, y, df


def build_pipeline():
    numeric_features = [
        "Credit_History_Score",
        "TotalIncome_Log",
        "LoanAmount_Risk",
        "Debt_Burden",
        "Dependents_Num",
    ]

    categorical_features = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", LoanFeatureEngineer(credit_weight=2.5)),
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    return full_pipeline, numeric_features, categorical_features


def train_and_evaluate():
    print("Loading data...")
    X, y, raw_df = load_and_preprocess_data("loan_data.csv")

    # Stratified Train-Test Split (80/20) strictly before fitting transformers
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")

    base_knn_pipeline, numeric_features, categorical_features = build_pipeline()

    # Baseline Model: Logistic Regression
    baseline_pipeline = Pipeline(
        steps=[
            ("feature_engineer", LoanFeatureEngineer(credit_weight=1.0)),
            ("preprocessor", base_knn_pipeline.named_steps["preprocessor"]),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    base_preds = baseline_pipeline.predict(X_test)
    base_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_acc = accuracy_score(y_test, base_preds)
    baseline_auc = roc_auc_score(y_test, base_proba)
    print(f"Logistic Regression Baseline - Test Accuracy: {baseline_acc:.4f}, ROC-AUC: {baseline_auc:.4f}")

    # Hyperparameter Grid Search for KNN
    print("Tuning KNN hyperparameters with 5-fold Stratified CV...")
    param_grid = {
        "classifier__n_neighbors": [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        "classifier__weights": ["uniform"],
        "classifier__metric": ["euclidean", "manhattan"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_knn_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    print(f"Best KNN Hyperparameters: {best_params}")
    print(f"Best 5-Fold CV Accuracy: {best_cv_score:.4f}")

    # Evaluate best model on Test Set
    test_preds = best_pipeline.predict(X_test)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, pos_label=1)
    recall = recall_score(y_test, test_preds, pos_label=1)
    f1 = f1_score(y_test, test_preds, pos_label=1)
    roc_auc = roc_auc_score(y_test, test_proba)
    cm = confusion_matrix(y_test, test_preds).tolist()
    report = classification_report(y_test, test_preds, output_dict=True)

    print("\n=== FINAL TEST EVALUATION ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))

    # Evaluate K-sensitivity curve for visualization in Streamlit
    k_range = list(range(1, 31, 2))
    k_accuracies_train = []
    k_accuracies_test = []
    
    transformer_part = Pipeline(best_pipeline.steps[:-1])
    X_train_trans = transformer_part.transform(X_train)
    X_test_trans = transformer_part.transform(X_test)

    metric_used = best_params.get("classifier__metric", "euclidean")
    weights_used = best_params.get("classifier__weights", "uniform")

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights_used, metric=metric_used)
        knn.fit(X_train_trans, y_train)
        k_accuracies_train.append(float(accuracy_score(y_train, knn.predict(X_train_trans))))
        k_accuracies_test.append(float(accuracy_score(y_test, knn.predict(X_test_trans))))

    # Get feature names after one-hot encoding
    preprocessor = best_pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names

    # Save artifacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_pipeline, "model/knn_pipeline.joblib")
    print("Saved pipeline to model/knn_pipeline.joblib")

    # Save training transformed data for neighbor inspection
    train_processed_df = pd.DataFrame(X_train_trans, columns=all_feature_names)
    train_processed_df["Loan_Status"] = y_train.values
    # Also attach original raw features for human readability in neighbor inspection
    X_train_raw = X_train.copy().reset_index(drop=True)
    X_train_raw["Loan_Status"] = y_train.values
    X_train_raw.to_csv("model/train_raw_with_target.csv", index=False)
    train_processed_df.to_csv("model/train_processed.csv", index=False)

    metrics_payload = {
        "best_params": best_params,
        "best_cv_score": float(best_cv_score),
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm,
            "classification_report": report,
        },
        "baseline_comparison": {
            "model": "Logistic Regression",
            "accuracy": float(baseline_acc),
            "roc_auc": float(baseline_auc),
        },
        "k_sensitivity": {
            "k_values": k_range,
            "train_accuracy": k_accuracies_train,
            "test_accuracy": k_accuracies_test,
        },
        "feature_names": all_feature_names,
    }

    with open("model/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("Saved metrics to model/metrics.json")


if __name__ == "__main__":
    train_and_evaluate()
