# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## 📌 Project Overview
A Machine Learning project to detect fraudulent credit card transactions using classification algorithms. This project addresses the real-world challenge of **class imbalance** in fraud detection and compares multiple ML models.

## 🎯 Problem Statement
Credit card fraud causes billions in losses every year. The challenge is:
- Only **~2% of transactions are fraudulent** (highly imbalanced dataset)
- Missing a fraud case (False Negative) is very costly
- We need **high Recall** without sacrificing too much Precision

## 🔧 Tech Stack
- **Language:** Python 3.8+
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
- **Technique:** SMOTE for handling class imbalance

## 📊 Models Compared
| Model | AUC-ROC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | ~0.85 | ~0.78 | ~0.80 | ~0.79 |
| Random Forest | ~0.95 | ~0.90 | ~0.88 | ~0.89 |
| **XGBoost** | **~0.97** | **~0.92** | **~0.91** | **~0.91** |

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### 3. Run the notebook
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

## 📁 Project Structure
```
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb   # Main project notebook
├── eda_plots.png                        # Exploratory Data Analysis charts
├── confusion_matrices.png               # Model evaluation plots
├── roc_curves.png                       # ROC curve comparison
├── feature_importance.png               # Feature importance chart
└── README.md                            # Project documentation
```

## 🔍 Key Steps
1. **Data Generation** - Simulated realistic fraud dataset with 10,000 transactions
2. **EDA** - Analyzed patterns between fraud and legitimate transactions
3. **Preprocessing** - Feature scaling + SMOTE to handle class imbalance
4. **Model Training** - Trained Logistic Regression, Random Forest, XGBoost
5. **Evaluation** - Compared using Precision, Recall, F1, AUC-ROC
6. **Prediction** - Real-time fraud prediction on new transactions

## 💡 Key Learnings
- **Recall is more important than Precision** in fraud detection — missing a fraud is worse than a false alarm
- **SMOTE** effectively handles class imbalance by generating synthetic minority samples
- **XGBoost outperforms** simpler models like Logistic Regression for complex fraud patterns
- **Feature engineering** (time of day, distance from home) significantly impacts model performance

## 👤 Author
**Your Name**
- LinkedIn: [linkedin.com/in/abhishek-mehta-890599239]
- Email: [abhishekaps999@gmial.com]

## 📄 License
MIT License
