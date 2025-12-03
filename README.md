# NIFTY Intraday Price Direction Prediction Using Machine Learning

This project uses ** NIFTY intraday OHLC data** to build machine learning models that **predict whether the next candle's closing price will go up or down**. Based on model predictions, **buy/sell trading signals** are generated and a **cumulative Profit & Loss (PnL)** is calculated.

---

## Project Objective

- Predict **next candle price direction (UP/DOWN)** using ML  
- Compare **multiple ML models**  
- Select the **best performing model**  
- Generate **automated buy/sell signals**  
- Simulate **cumulative trading PnL** on test data  

---



---


## Target Label Generation

A new column `target` is created:
- `target = 1` → If **next close > current close**
- `target = 0` → If **next close < current close**

This converts the problem into a **supervised binary classification task**.

---

##  Machine Learning Models Used

1. Logistic Regression  
2. Random Forest  
3. Gradient Boosting  
4. Support Vector Machine (SVM)  

The **best model is chosen based on highest Accuracy**.

---

## Model Evaluation Metrics

- Accuracy  
- Precision  
- Recall
- F1 Score

Metrics for all models are printed during execution.

---

## Trading Signal Generation

On the **test dataset**, a new column `model_call` is created:
- Prediction `1` → `"buy"`
- Prediction `0` → `"sell"`

---

##  Profit & Loss (PnL) Calculation

A cumulative `model_pnl` column is calculated:
- `"buy"` → subtract `close`
- `"sell"` → add `close`

PnL is updated **row by row in time order**.

---

## Final Output

Final test file is saved at:
```
outputs/test_with_predictions_and_pnl.csv
```

Columns include:
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `target`
- `Predicted`
- `model_call`
- `model_pnl`

---

## Setup Instructions

### Create Virtual Environment

```bash
python -m venv venv
```

**Activate:**

**Windows**
```bash
venv\Scripts\activate
```

**Linux/Mac**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python -m src.main
```









## 📝 License

This project is for educational purposes only.
