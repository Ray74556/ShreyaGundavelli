# Credit Card Fraud Detection — Custom Neural Network

A one-hidden-layer multilayer perceptron (MLP) built **from scratch with NumPy**
to classify fraudulent credit-card transactions. The dataset is severely
imbalanced: approximately 0.17% of transactions are fraud (492 of 284,807).

## Why Build an MLP From Scratch?

Libraries such as PyTorch and TensorFlow provide efficient neural-network
implementations. This project instead implements the core training mechanics
directly with NumPy to demonstrate understanding of forward propagation,
backpropagation, activation functions, binary cross-entropy loss, and
mini-batch gradient descent.

## Architecture

30 input features -> 64 hidden neurons (ReLU) -> 1 output neuron (Sigmoid)

## Dataset

This project uses the
[ULB Machine Learning Group credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features: anonymized PCA components (`V1`–`V28`), `Time`, and `Amount`
- Binary target: `Class` (`0` = legitimate, `1` = fraud)

## Methodology

1. Remove duplicate records.
2. Create a stratified train/test split from the original dataset to preserve
   the real fraud rate in the test set.
3. Randomly undersample legitimate transactions in the **training set only**.
4. Fit `StandardScaler` on training features only, then transform both training
   and test features with that fitted scaler.
5. Train the custom MLP using mini-batch gradient descent.
6. Compare three epoch and learning-rate configurations.
7. Evaluate the final model on the untouched test set using precision, recall,
   F1-score, and a confusion matrix.

## Project Structure

```text
fraud-detection-nn/
├── run_experiments.py       # Entry point; runs all experiments
├── src/
│   ├── data_loader.py       # Downloads dataset with kagglehub
│   ├── neural_network.py    # From-scratch NumPy MLP
│   ├── experiment_log.py    # Records configuration and evaluation metrics
│   └── visualize.py         # Generates figures
├── requirements.txt
└── .gitignore
```

## Running Locally

```bash
pip install -r requirements.txt
python run_experiments.py
```

The script downloads the dataset through `kagglehub` on the first run, executes
three experiments, writes results to `experiment_log.txt`, and saves:

- `fig1_nn_arch.png` — Neural-network architecture diagram
- `fig2_class_dist.png` — Class distribution before and after training-set undersampling
- `fig3_metrics_compare.png` — Precision, recall, F1, and accuracy across experiments
- `fig4_confusion_matrix.png` — Confusion matrix for the best F1-score experiment

## Preliminary Results: Balanced Test Split

| Experiment | Epochs | Learning Rate | Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| 1 — Best F1 | 100 | 0.001 | 0.60 | 0.6389 | 0.4842 | 0.5509 |
| 2 | 500 | 0.005 | 0.90 | 0.4143 | 0.6105 | 0.4936 |
| 3 — Highest Recall | 750 | 0.009 | 0.90 | 0.2913 | 0.7789 | 0.4241 |

> All metrics were calculated on a held-out test set retaining the original
0.167% fraud prevalence. Thresholds were selected during exploratory analysis
to demonstrate the precision–recall tradeoff.

## Limitations

- The dataset contains anonymized features, so individual fraud-risk drivers
  cannot be interpreted in business terms.
- This project is an offline classification experiment, not a real-time or
  production-deployed fraud-detection system.
- Future work could compare the custom MLP with logistic regression, add
  precision-recall curves, and use cross-validation for hyperparameter tuning.

## Skills Demonstrated

Python, NumPy, pandas, scikit-learn, neural-network fundamentals,
data preprocessing, class-imbalance handling, model evaluation,
Matplotlib, and Seaborn.