"""
Custom multi-layer perceptron for credit card fraud detection, implemented from scratch
with NumPy, using forward pass backward pass and sigmoid cross-entropy loss  are all handwritten.

Architecture: 30 input features -> 64 hidden neurons (ReLU) -> 1 output (sigmoid)

"""

import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sk_train_test_split

def load_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    return df

class NeuralNetwork:
    def __init__(self, df = pd.DataFrame):
        self.df = df
        self.X = df.drop(columns =['Class'])
        self.y = df['Class']
        self.epochs = None
        self.learning_rate = None

    """ Drop duplicates, random-undersample the majority class down to the 
    fraud, and standardize features."""
    def preprocess(self):
        self.df.drop_duplicates(inplace=True)

        fraud = self.df[self.df["Class"] == 1]
        normal = self.df[self.df["Class"] == 0]

        normal_undersample = normal.sample(n=len(fraud), random_state=42)
        self.df = pd.concat([fraud, normal_undersample])
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.X = self.df.drop(columns =['Class']).to_numpy()
        self.y = self.df["Class"].to_numpy()

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def train_test_split(self, test_size = 0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = sk_train_test_split(self.X, self.y, test_size=test_size, random_state=42, stratify=self.y)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def check_split(self):
        print("Train shape: ", self.X_train.shape)
        print("Test shape: ", self.X_test.shape)
        print("Train fraud count: ", self.y_train.shape)
        print("Test fraud count: ", self.y_test.shape)

    def initialize_weights(self):
        input_size = self.X.shape[1]
        hidden_size =  64
        output_size = 1

        # initialize weights and biases for input and hidden layer
        # He initialization for ReLU hidden Layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1  = np.zeros(hidden_size)

        # initialize weights and biases for hidden and output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.loss_hist = []

        # test
        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)

    def forward_propagation(self, X_batch):
        self.Z1 = np.dot(X_batch, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-np.clip(self.Z2, -500, 500)))
        return self.A2

    def loss_func(self, y_batch):
        s = y_batch.shape[0]
        y_batch = y_batch.reshape(-1, 1)  # match self.A2's shape: (n, 1)
        self.loss = (-1 / s) * np.sum(
            (y_batch * np.log(self.A2 + 1e-8)) + ((1 - y_batch) * np.log(1 - self.A2 + 1e-8))
        )

    def _compute_loss(self, y_true, y_pred_proba):
        """Stateless loss computation -- doesn't touch self.Z1/A1/etc.,
        safe to call for validation without corrupting training state."""
        y_true = y_true.flatten()
        y_pred_proba = y_pred_proba.flatten()
        s = y_true.shape[0]
        return (-1 / s) * np.sum(
            y_true * np.log(y_pred_proba + 1e-8) + (1 - y_true) * np.log(1 - y_pred_proba + 1e-8)
        )

    def backward_propagation(self, X_batch, y_batch):
        training = X_batch.shape[0]

        self.dZ2 = self.A2 - y_batch.reshape(-1, 1)
        self.dW2 = 1 / training * np.dot(self.A1.T, self.dZ2)
        self.db2 = 1 / training * np.sum(self.dZ2, axis=0)

        self.dA1 = np.dot(self.dZ2, self.W2.T)
        self.dZ1 = self.dA1 * (self.Z1 > 0)

        self.dW1 = 1 / training * np.dot(X_batch.T, self.dZ1)
        self.db1 = 1 / training * np.sum(self.dZ1, axis=0)

    def grad_descent(self, learning_rate=0.01):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def train(self, epochs=1000, learning_rate=0.01, batch_size=256, X_val=None, y_val=None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        n = self.X_train.shape[0]
        self.val_loss_hist = []
        self.val_acc_hist = []

        for epoch in range(epochs):
            indicies = np.random.permutation(n)
            X_shuffled = self.X_train[indicies]
            y_shuffled = self.y_train[indicies]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.forward_propagation(X_batch)
                self.loss_func(y_batch)
                self.backward_propagation(X_batch, y_batch)
                self.grad_descent(learning_rate)

                epoch_loss += self.loss
                n_batches += 1

            self.loss = epoch_loss / n_batches
            self.loss_hist.append(self.loss)

            # evaluate on the held-out set every epoch, without training on it
            if X_val is not None and y_val is not None:
                val_preds_proba = self.forward_propagation(X_val)
                val_loss = self._compute_loss(y_val, val_preds_proba)  # see note below
                val_acc = np.mean((val_preds_proba >= 0.5).flatten() == y_val.flatten())
                self.val_loss_hist.append(val_loss)
                self.val_acc_hist.append(val_acc)

            if epoch % 100 == 0:
                msg = f" Epoch {epoch + 1}/{epochs}: Train Loss: {self.loss:.6f}"
                if X_val is not None:
                    msg += f" | Val Loss: {self.val_loss_hist[-1]:.6f} | Val Acc: {self.val_acc_hist[-1]:.4f}"
                print(msg)

    def predict_and_eval(self, X, y, threshold=0.3):
        """Predict X/y and return metrics"""

        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = 1 / (1 + np.exp(-np.clip(Z2, -500, 500)))

        predictions = (A2 >= threshold).astype(int).flatten()
        y = y.flatten()

        accuracy = np.sum(predictions == y) / len(predictions)

        TP = np.sum((predictions == 1) & (y == 1))
        TN = np.sum((predictions == 0) & (y == 0))
        FP = np.sum((predictions == 1) & (y == 0))
        FN = np.sum((predictions == 0) & (y == 1))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "CM": [[TN, FP], [FN, TP]],
        }

if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
