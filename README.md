# 🛡️ FinShield — AI-Powered Payment Transaction Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Status-Capstone%20Prototype-green.svg)](#)

---

## 📌 1. Project Overview

**FinShield** is an end-to-end Machine Learning anomaly detection system designed to identify fraudulent credit card transactions in near real-time. Built as a capstone project for **Learn Depth Academy LLP**, FinShield implements a robust data-cleaning and feature-scaling pipeline, models class imbalance directly using cost-sensitive learning (`class_weight="balanced"`), systematically benchmarks three candidate classifiers, and selects the optimal model using **ROC-AUC** as the primary decision metric.

The validated model and preprocessing transformers are persisted and served through an interactive **Streamlit web application**, providing automated fraud probability estimation, risk tier categorization, and single-transaction simulation.

> **Disclaimer:** FinShield is an educational machine learning prototype developed for academic demonstration. It is not an enterprise-grade banking production system.

---

## ⚠️ 2. Problem Statement

Financial fraud poses a severe, multi-billion-dollar global challenge. Modern automated payment gateways process millions of transactions daily, but fraudulent attempts constitute an infinitesimal fraction of total volume (typically under 0.2%).

Standard machine learning models trained on naive accuracy fail catastrophically in this regime: a dummy classifier that labels every single transaction as "genuine" achieves **>99.8% accuracy**, yet fails to detect a single fraudulent charge. 

The core challenge is balancing:
- **False Negatives (Missed Fraud):** Direct monetary loss, chargeback liabilities, and compromised customer security.
- **False Positives (False Alarms):** Unnecessary card freezes, merchant friction, and customer dissatisfaction.

A practical fraud system must optimize discriminatory power across all probability thresholds rather than optimizing raw accuracy.

---

## 🎯 3. Project Objective

The primary objectives of the FinShield project are:
1. Conduct extensive Exploratory Data Analysis (EDA) on anonymized credit card transaction records to identify signatures of anomalous behavior.
2. Build a leak-free preprocessing pipeline handling duplicate records, stratified data partitioning, and standardized numerical feature scaling.
3. Train and benchmark three classification architectures (**Logistic Regression**, **Decision Tree**, and **K-Nearest Neighbors**) under severe class imbalance.
4. Select the final model strictly using **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve) to prioritize fraud discrimination capability.
5. Serialize the trained pipeline (`joblib`) and deploy a real-time **Streamlit** user interface with pre-built demo scenarios for interactive evaluation.

---

## ✨ 4. Key Features

- **Automated Data Cleaning & Deduplication:** Removes exact duplicate entries without information leakage.
- **Leak-Free Scaling:** Fits `StandardScaler` strictly on training folds to prevent data snooping.
- **Cost-Sensitive Learning:** Employs balanced class weights (`class_weight="balanced"`) to penalize false negatives proportionally without synthetic oversampling noise.
- **Objective Metric Selection:** Explicit selection criterion grounded in **ROC-AUC** to maximize true positive rate while penalizing false positive rate.
- **Streamlit Web Dashboard:** Interactive UI featuring instant probability calculation, risk stratification (**Low**, **Medium**, **High**), and instant pre-filled demo buttons for genuine and fraudulent transactions.
- **Modern Packaging with uv:** Dependency management and execution isolated via Astral's fast `uv` workspace manager.

---

## 📊 5. Dataset Description & Characteristics

FinShield uses the benchmark **Credit Card Fraud Detection** dataset published by the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB).

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Total Original Transactions** | `284,807` | Collected over a 2-day period in September 2013 |
| **Genuine Transactions (`Class 0`)** | `284,315` | **99.827%** of dataset |
| **Fraudulent Transactions (`Class 1`)** | `492` | **0.173%** of dataset (Extreme Imbalance: ~578:1 ratio) |
| **Total Columns** | `31` | 30 input features + 1 binary target |
| **PCA Features (`V1` to `V28`)** | 28 features | Numerical features resulting from Principal Component Analysis (confidentiality protection) |
| **Original Features** | `Time`, `Amount` | `Time`: elapsed seconds from first transaction; `Amount`: transaction value in EUR |
| **Target Variable** | `Class` | `0` = Genuine / Normal, `1` = Fraudulent / Anomalous |
| **Missing Values** | `0` | Clean numerical matrix without NaNs |
| **Duplicate Rows Identified** | `1,081` | Removed during preprocessing |

### Dataset Citation
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. In IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

---

## 🛠️ 6. Technology Stack

- **Language:** Python 3.14
- **Environment & Package Manager:** [uv](https://github.com/astral-sh/uv) (Astral)
- **Data Manipulation:** [pandas](https://pandas.pydata.org/) (v3.0.6), [numpy](https://numpy.org/) (v2.5.3)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (v1.9.1)
- **Data Visualization:** [matplotlib](https://matplotlib.org/) (v3.11.2), [seaborn](https://seaborn.pydata.org/) (v0.13.2)
- **Interactive Application:** [Streamlit](https://streamlit.io/) (v1.64.0)
- **Model Serialization:** [joblib](https://joblib.readthedocs.io/) (v1.6.0)
- **Notebook Environment:** [JupyterLab / ipykernel](https://jupyter.org/)

---

## 🔄 7. Machine Learning Workflow

```mermaid
flowchart TD
    A[Raw Dataset: creditcard.csv<br/>284,807 rows] --> B[Data Inspection & Cleaning<br/>Drop 1,081 duplicates -> 283,726 rows]
    B --> C[Exploratory Data Analysis<br/>Imbalance, Hour feature, Correlation]
    C --> D[Stratified Train-Test Split<br/>80% Train: 226,980 | 20% Test: 56,746]
    D --> E[Feature Scaling<br/>StandardScaler fit on Train, applied to Test]
    E --> F[Model Training with class_weight='balanced']
    F --> F1[Logistic Regression]
    F --> F2[Decision Tree]
    F --> F3[KNN]
    F1 & F2 & F3 --> G[Model Evaluation & Comparison<br/>Accuracy, Precision, Recall, F1, ROC-AUC]
    G --> H[Final Selection: Logistic Regression<br/>Highest ROC-AUC: 0.9684]
    H --> I[Artifact Persistence<br/>model.pkl + scaler.pkl]
    I --> J[Streamlit Interactive Web Application<br/>Real-time scoring, Risk tiers, Demos]
```

---

## 📈 8. Exploratory Data Analysis (EDA) Summary

Key findings uncovered during EDA in `notebooks/01_EDA.ipynb`:

1. **Extreme Target Imbalance:** Fraud represents only **0.173%** of all transactions.
2. **Transaction Amount Patterns:** 
   - Genuine transactions exhibit a right-skewed distribution with a mean of **€88.35** (median €22.00, maximum €25,691.16).
   - Fraudulent transactions, while occasionally high, predominantly feature small-to-moderate sums (mean ~€122) designed to slip past authorization thresholds.
3. **Temporal Behavior:** Converting raw elapsed seconds into a 24-hour cycle (`(Time // 3600) % 24`) reveals that fraudulent activity remains sustained during early morning hours (01:00–05:00) when genuine transaction volumes dip significantly.
4. **Significant Feature Correlations:**
   - **Strong Negative Correlation with Fraud:** `V14` (-0.302), `V17` (-0.326), `V12` (-0.260), and `V10` (-0.216). Unusually low negative values on these components serve as primary indicators of fraud.
   - **Strong Positive Correlation with Fraud:** `V11` (+0.155), `V4` (+0.133), and `V2` (+0.091).

---

## ⚙️ 9. Data Preprocessing & Class Imbalance Handling

### Preprocessing Steps (`src/preprocess.py`):
1. **Deduplication:** Dropped `1,081` duplicate transactions, yielding `283,726` unique transaction records.
2. **Stratified Split:** Split features and target using an 80/20 stratified split (`stratify=y`, `random_state=42`), preserving the exact fraud proportion across subsets:
   - **Training Set (`X_train`):** `226,980` transactions (397 fraud cases)
   - **Testing Set (`X_test`):** `56,746` transactions (95 fraud cases)
3. **Feature Standardization:** `V1`–`V28` were already zero-centered and scaled via PCA. `StandardScaler` was applied exclusively to `Time` and `Amount`. The scaler was fitted solely on `X_train` and subsequently transformed `X_test` to prevent data snooping.

### Class Imbalance Handling:
Rather than synthetically generating borderline samples (which can introduce noise in high-dimensional PCA spaces), class imbalance was handled mathematically via **cost-sensitive learning**:
- `class_weight="balanced"` was configured for Logistic Regression and Decision Tree.
- In scikit-learn, this assigns inverse class weights proportional to class frequencies:
  $$\text{weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_{\text{samples}, c}}$$
- For the minority fraud class, this scales up the penalty for misclassification by approximately **578×**, ensuring the optimizer prioritizes high recall for anomalous events.

---

## 🤖 10. Models Benchmarked

Three distinct algorithms were implemented and trained under controlled conditions (`random_state=42`):

1. **Logistic Regression:**
   - Hyperparameters: `class_weight="balanced"`, `max_iter=1000`, `solver="lbfgs"`
   - Rationale: High-speed linear baseline offering smooth calibrated probability scores and convex convergence.
2. **Decision Tree Classifier:**
   - Hyperparameters: `class_weight="balanced"`, `max_depth=8`, `random_state=42`
   - Rationale: Captures non-linear feature interactions and orthogonal decision boundaries while pruning depth to prevent overfitting.
3. **K-Nearest Neighbors (KNN):**
   - Hyperparameters: `n_neighbors=5`, Euclidean distance
   - Rationale: Non-parametric instance-based classifier evaluating local cluster density.

---

## 🏆 11. Model Comparison & Selection

All models were evaluated on the held-out test set (`56,746` transactions, including `95` fraud instances).

### Actual Evaluation Results Table:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **0.9748** | **0.0553** | **0.8737** | **0.1040** | **0.9684** |
| **Decision Tree** | 0.9913 | 0.1374 | 0.8000 | 0.2346 | 0.9074 |
| **KNN** | 0.9994 | 0.9420 | 0.6842 | 0.7927 | 0.8841 |

### Final Model Selection Rationale:

The project explicitly establishes **ROC-AUC** as the primary model-selection criterion.

- **Why Logistic Regression is Selected:**
  - Logistic Regression achieved the highest ROC-AUC score (**0.9684**), outperforming Decision Tree (0.9074) and KNN (0.8841).
  - It delivered the highest recall (**87.37%**), catching 83 out of 95 fraud cases in the test set. In financial risk systems, missing a fraudulent transaction (False Negative) is substantially more costly than investigating a legitimate flagged transaction (False Positive).
- **Why KNN was NOT Selected:**
  - While KNN achieved a seemingly high precision (0.9420) and F1-score (0.7927) at the default 0.5 threshold, it achieved the lowest ROC-AUC (**0.8841**) and lowest recall (**68.42%**), missing nearly one-third of all fraud attempts (30 missed fraudulent transactions).
  - Additionally, KNN suffers from $O(N \cdot D)$ inference latency, making it impractical for streaming payment systems.
- **Why Accuracy is Misleading:**
  - All three models achieve >97% accuracy. Even an uninformative dummy model predicting all transactions as genuine would achieve 99.83% accuracy. Thus, accuracy is discarded as an operational metric.

**Decision:** **Logistic Regression** is the chosen final model.

---

## 🎯 12. Final Model Evaluation & Confusion Matrix

### Evaluation Metrics (Logistic Regression on Test Set):
- **Accuracy:** `97.48%`
- **Precision:** `5.53%`
- **Recall (Sensitivity):** `87.37%`
- **F1-Score:** `10.40%`
- **ROC-AUC:** `96.84%`

### Confusion Matrix:

```
                  Predicted Genuine (0)    Predicted Fraud (1)
Actual Genuine (0)        55,233                   1,418          [56,651]
Actual Fraud (1)             12                      83           [    95]
```

### Interpretation:
- **True Negatives (TN = 55,233):** Genuine transactions correctly cleared without friction.
- **False Positives (FP = 1,418):** Legitimate transactions flagged for secondary verification (e.g., OTP or SMS challenge).
- **False Negatives (FN = 12):** Undetected fraudulent transactions (only 12 out of 95 missed).
- **True Positives (TP = 83):** Fraudulent transactions intercepted and blocked.

### Why ROC-AUC Matters:
ROC-AUC measures the probability that the classifier will rank a randomly chosen fraudulent transaction higher than a randomly chosen genuine transaction across all possible operational cutoffs. An AUC of **0.9684** demonstrates near-optimal discriminatory ranking, allowing risk teams to dynamically adjust the decision threshold based on shifting fraud-to-friction cost tolerances.

---

## 💻 13. FinShield Streamlit Application

The interactive web application (`app/streamlit_app.py`) provides an operational interface for fraud risk assessment:

- **Input Controls:** Sliders/number inputs for transaction `Time`, `Amount`, and PCA features `V1`–`V28`.
- **Feature Pipeline Synchronization:** Automatically calculates the `Hour` feature (`(Time // 3600) % 24`), applies the saved `StandardScaler` to `Time` and `Amount`, and guarantees exact feature column ordering matching `model.feature_names_in_`.
- **Risk Stratification:**
  - **High Risk:** Fraud Probability $\ge 80\%$
  - **Medium Risk:** Fraud Probability between $50\%$ and $79\%$
  - **Low Risk:** Fraud Probability $< 50\%$
- **One-Click Demo Buttons:**
  - **Genuine Demo:** Sets typical genuine transaction parameters (`Time = 50,000`, `Amount = 100`, `V1–V28 = 0.0`).
  - **Fraud Demo:** Sets anomalous parameters observed during EDA (`Time = 60,000`, `Amount = 150`, `V10 = -6.2`, `V12 = -5.8`, `V14 = -7.1`, `V17 = -6.4`, all other $V = 0.0$).

---

## 📂 14. Project Folder Structure

```
capstone-fraud-detection/
├── .venv/                      # Project-local virtual environment (managed by uv)
├── app/
│   └── streamlit_app.py        # Streamlit web application dashboard
├── assets/
│   ├── plots/                  # Extracted EDA, ROC curve, and confusion matrix charts
│   └── screenshots/            # UI screenshots and application captures
├── data/
│   └── creditcard.csv          # Credit Card Fraud Detection dataset (Kaggle/ULB)
├── models/
│   ├── model.pkl               # Persisted final Logistic Regression model
│   └── scaler.pkl              # Persisted StandardScaler transformer
├── notebooks/
│   └── 01_EDA.ipynb            # Jupyter notebook: EDA, training, and benchmarking
├── src/
│   ├── __init__.py             # Package marker
│   ├── preprocess.py           # Data deduplication, splitting, and scaling logic
│   ├── train.py                # Standalone script to train and serialize model
│   └── evaluate.py             # Model evaluation and classification report script
├── FinShield_Presentation.pptx # Capstone presentation slides (12 slides)
├── pyproject.toml              # Project metadata and dependency definitions
├── requirements.txt            # Pinned requirements file
├── uv.lock                     # Deterministic dependency lockfile
└── README.md                   # Complete project documentation
```

---

## 🚀 15. Installation & Execution Guide

### Prerequisites
- Python `>= 3.14`
- [uv](https://docs.astral.sh/uv/) package manager installed:
  ```bash
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Step 1: Clone and Setup Environment
Navigate to the project root directory and synchronize dependencies:

```powershell
cd d:\HDD\HDD\ENGINEERING\06_PROJECTS\LearnDepthInternship\Project(3)_Final_Capstone_Project\capstone-fraud-detection

# Sync dependencies using uv
uv sync
```

Alternatively, using standard pip:
```powershell
pip install -r requirements.txt
```

---

### Step 2: Running the EDA Notebook
To explore the analysis and reproduce visualizations:

```powershell
uv run jupyter lab
# Or:
uv run jupyter notebook
```
Open and run `notebooks/01_EDA.ipynb`.

---

### Step 3: Training the Model
To re-train the Logistic Regression model and save artifacts to `models/`:

```powershell
cd src
uv run python train.py
```
*Output: `Model trained and saved successfully.`*

---

### Step 4: Evaluating the Model
To compute test set performance, confusion matrix, and ROC-AUC:

```powershell
cd src
uv run python evaluate.py
```

---

### Step 5: Launching the Streamlit Application
To launch the interactive fraud detection interface:

```powershell
cd app
uv run streamlit run streamlit_app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## ⚠️ 16. Limitations

1. **Synthetic Feature Anonymity:** Due to confidentiality, features `V1`–`V28` are anonymized PCA projections. True business context (e.g., merchant category, IP location, cardholder age) is inaccessible.
2. **Fixed Threshold:** The prototype evaluates decisions using a default threshold of $p = 0.5$. In commercial banking, this threshold is continuously adjusted dynamically based on transaction value and cost matrix considerations.
3. **Data Drift & Evolving Fraud Patterns:** Fraud tactics evolve dynamically. Static models trained on historical data require automated retraining pipelines and concept-drift monitoring.
4. **Educational Prototype:** This system is not hardened for concurrent production scale, PCI-DSS compliance, or encrypted tokenized payload ingestion.

---

## 🔮 17. Future Scope

- **Advanced Gradient Boosting:** Benchmark modern gradient boosting frameworks (XGBoost, LightGBM, CatBoost) with focal loss functions.
- **Cost-Sensitive Threshold Tuning:** Implement cost-utility curve analysis to optimize the probability cutoff against monetary chargeback vs. customer intervention costs.
- **Model Explainability:** Integrate SHAP (SHapley Additive exPlanations) or LIME to present human investigators with top contributing risk factors for each flagged transaction.
- **Real-Time Streaming Architecture:** Connect the scoring engine to Apache Kafka or AWS Kinesis for sub-millisecond transaction stream processing.
- **Unsupervised Anomaly Detection:** Deploy Isolation Forests, Autoencoders, or One-Class SVMs to identify novel, zero-day fraud mechanisms.

---

## 📌 18. Conclusion

The **FinShield** project provides a comprehensive, mathematically grounded solution to credit card fraud classification under extreme class imbalance. By establishing **ROC-AUC** as the rigorous selection criterion, the project prioritized fraud detection recall (87.37%) and ranked discrimination (AUC 0.9684) over misleading accuracy metrics. With an automated preprocessing pipeline, serialized models, and an interactive Streamlit application, FinShield successfully satisfies all academic and technical requirements of the Learn Depth Academy capstone program.

---

## 📚 19. References

- ULB Machine Learning Group: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Scikit-learn Documentation: [Model Evaluation & Cost-Sensitive Classification](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Streamlit Documentation: [Streamlit Application Framework](https://docs.streamlit.io/)
- Dal Pozzolo, A., et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE CIDM.
