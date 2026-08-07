# GridSecure — Electricity Theft Detection System
## Methodology, Evaluation Matrices & Technical Approach Rationale

---

### 1. Technical Approach & Methodology Rationale

Electricity theft detection presents a unique challenge in data analytics and machine learning:

#### A. Class Imbalance Handling
- **Challenge**: In smart meter datasets, electricity theft cases represent only **~8.5%** of total consumers (3,615 theft cases out of 42,372 total consumers).
- **Our Approach**: 
  1. We apply a **Stratified 80/20 Train-Test Split** (`train_test_split(..., stratify=y)`). This ensures both training and testing datasets retain the exact same 8.5% theft ratio so the model isn't evaluated on biased splits.
  2. We configure class weighting (`class_weight='balanced'`) during classifier fitting. This penalizes false negative theft misclassifications more heavily during training, forcing the model to pay attention to rare theft patterns.

#### B. Feature Engineering & Scaling
- **Our Approach**:
  1. **Metadata Isolation**: Categorical identifiers (`CONS_NO`, `Locality`, `City`, `State`) are separated so the model learns purely from consumption behavior rather than geographic bias.
  2. **Categorical Encoding**: Text features (`Urban_Rural`, `Consumer_Type`) are mapped to numeric integers using `LabelEncoder`.
  3. **Standard Scaling**: Continuous variables (`Avg_Consumption`, `Zero_Consumption_Days`, `Behavioural_Anomaly_Score`) are normalized using `StandardScaler` so high-magnitude features don't drown out smaller normalized scores.

---

### 2. Evaluation Metrics & Matrices Explained

Standard classification accuracy is misleading in imbalanced datasets (a dummy model predicting all "Normal" would achieve 91.5% accuracy but detect 0% theft). Therefore, we evaluate using multiple complementary matrices:

#### A. Confusion Matrix Analysis (`docs/confusion_matrices.png`)
The confusion matrix tracks four key outcomes:
- **True Positives (TP)**: Actual thieves correctly identified by the model (Enables utility revenue recovery).
- **False Positives (FP)**: Normal consumers mistakenly flagged as thieves (Causes unnecessary field inspection costs).
- **True Negatives (TN)**: Normal consumers correctly classified (Avoids unnecessary dispatches).
- **False Negatives (FN)**: Actual thieves missed by the model (Leads to unrecovered power loss).

#### B. Key Evaluation Formulas:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{(How trustworthy is a theft alert?)}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(What fraction of total thieves did we catch?)}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad \text{(Harmonic mean balancing Precision & Recall)}$$

#### C. ROC-AUC Curves (`docs/roc_curves.png`)
The **Receiver Operating Characteristic (ROC)** curve plots True Positive Rate against False Positive Rate across all decision thresholds. The **Area Under Curve (AUC)** measures overall ranking quality:
- **AUC = 1.0**: Perfect classification.
- **AUC = 0.5**: Random guessing.
- Our models achieve **AUC ~0.767**, proving strong discriminative capability.

---

### 3. Model Performance & Selection Rationale

We evaluated Member 3's 3 classifiers on the 42,372 consumer dataset:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Evaluation Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Best)** | **0.8838** | **0.3321** | **0.3582** | **0.3446** | **0.7670** | **Highest Accuracy, Precision, F1-Score & ROC-AUC.** |
| Logistic Regression | 0.7467 | 0.1926 | 0.6169 | 0.2935 | 0.7515 | High recall but excessive false alarms (low precision). |
| Decision Tree | 0.7292 | 0.1836 | 0.6307 | 0.2844 | 0.7296 | Lower overall accuracy and higher variance. |

#### Why Random Forest Outperformed Other Models:
1. **Handles Non-Linear Relationships**: Electricity consumption patterns non-linearly interact (e.g., zero consumption is suspicious *only* when combined with high anomaly scores). Logistic Regression assumes linear boundaries and fails to capture these complex interactions.
2. **Reduces Overfitting via Ensembling**: Single Decision Trees easily overfit noise in individual daily readings. Random Forest averages predictions across 100 decision trees trained on random feature subsets, giving robust generalizability.

---

### 4. Standalone Execution (No Frontend Required)

You can run the full project standalone directly from terminal without needing any web browser or frontend server:

```bash
python3 src/main.py
```

#### What `src/main.py` Does Standalone:
1. Loads dataset (`data/data.csv`).
2. Cleans missing values and generates `docs/EDA_Report.csv`.
3. Trains all 3 classifiers and exports `docs/Model_Comparison_Table.csv`.
4. Saves best Random Forest model (`models/gridsecure_best_model.pkl`) and visual plots (`docs/`).
5. Runs a sample consumer theft risk prediction and outputs probability %, risk category (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and actionable risk factors to the terminal.
