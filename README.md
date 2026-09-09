# CrediFlux | Next.js 15 & KNN Loan Prediction Intelligence Engine

A modern, full-stack **Next.js 15** web application paired with a high-performance **FastAPI Machine Learning backend** trained on the **Kaggle Loan Prediction Problem Dataset** (`altruistdelhite04/loan-prediction-problem-dataset`).

---

## 🌟 Key Capabilities

1. **Next.js 15 Full-Stack Web Application (`frontend/`)**:
   - **Modern Aesthetic**: Rich Vanilla CSS design system with CSS custom properties, dark/light theme switcher, glassmorphic cards, radiant neon gradients, and Google Fonts (`Outfit` & `JetBrains Mono`).
   - **Loan Decision Simulator**: Interactive sliders and selectors for all applicant attributes with quick-fill presets (*Strong*, *Borderline*, *High-Risk*).
   - **Decision Banner & Financial Indicators**: Instant verdict (Approved/Rejected), dynamic confidence bar, household income, Debt-to-Income (DTI) % with leverage tier, and monthly principal.
   - **KNN Neighbor Inspector**: Real-time inspection of the top $K$ nearest historical applicant records from the training data, showing Euclidean distances, similarity percentages, and ground-truth loan approval outcomes.
   - **Dataset Intelligence & EDA**: Visual summaries of credit history impact, property area distributions, and searchable/filterable historical records.
   - **Model Diagnostics**: Confusion Matrix heatmap, $K$-sensitivity curve ($K=1 \dots 31$), and baseline comparison against Logistic Regression.
   - **Batch Loan Underwriting**: CSV template download, drag-and-drop file upload, automated batch scoring, and one-click enriched CSV export.

2. **Domain-Aware KNN Machine Learning Pipeline**:
   - **Features**: Weighted `Credit_History_Score` ($2.5\times$ weight), log household income `TotalIncome_Log`, bounded `LoanAmount_Risk` to prevent small loan penalty outliers, `Debt_Burden` (EMI/Income %), and `Dependents_Num`.
   - **Test Accuracy**: **84.55%**
   - **Precision**: **82.35%**
   - **Recall**: **98.82%** (84/85 approved identified)
   - **F1 Score**: **89.84%**
   - **ROC-AUC**: **81.35%**

---

## 🚀 How to Run the Application

### Quick Launch (Both Backend & Frontend)
In PowerShell:
```powershell
.\start_nextjs.ps1
```

### Or Run Services Individually:

**1. Start the FastAPI ML Backend (Port 8000):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**2. Start the Next.js Frontend (Port 3000):**
```powershell
cd frontend
npm run dev -- -p 3000
```

Open your browser to: **`http://localhost:3000`**

*(Note: The Next.js frontend automatically proxies `/api/*` requests to the FastAPI backend on port 8000 via internal rewrites).*

---

## 📂 Project Architecture

```
loan prediction knn/
├── api.py                           # FastAPI REST API serving model inference & EDA
├── feature_engineering.py           # Domain-aware scikit-learn feature transformer
├── train_knn.py                     # Training & CV hyperparameter tuning script
├── loan_data.csv                    # Kaggle Loan Prediction Dataset (614 rows)
├── start_nextjs.ps1                 # Single-click PowerShell startup script
├── model/
│   ├── knn_pipeline.joblib          # Serialized scikit-learn pipeline
│   ├── metrics.json                 # Evaluation metrics & K-sensitivity points
│   ├── train_processed.csv          # Normalized feature dataset
│   └── train_raw_with_target.csv    # Raw training records with labels
└── frontend/                        # Next.js 15 Web Application
    ├── app/
    │   ├── globals.css              # Custom Vanilla CSS Design System
    │   ├── layout.tsx               # Root layout & Google Fonts
    │   └── page.tsx                 # Main dashboard layout
    ├── components/
    │   ├── Header.tsx               # Header with dark/light mode toggle
    │   ├── GlobalMetrics.tsx        # Top performance metric cards
    │   ├── SimulatorTab.tsx         # Loan simulator & KNN Neighbor Inspector
    │   ├── EdaTab.tsx               # Dataset analytics & interactive table
    │   ├── DiagnosticsTab.tsx       # Confusion matrix & K-curve diagnostics
    │   └── BatchTab.tsx             # Bulk CSV evaluation & export
    ├── next.config.ts               # Proxy rewrites to FastAPI backend
    └── package.json                 # Next.js & React dependencies
```
