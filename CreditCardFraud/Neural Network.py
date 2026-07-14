"""
Custom multi-layer perceptron for credit card fraud detection, implemented from scratch
with NumPy, using forward pass backward pass and sigmoid cross-entropy loss  are all handwritten.

Architecture: 30 input features -> 64 hidden neurons (ReLU) -> 1 output (sigmoid)

"""

import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        self.df = self.df.drop_duplicates(inplace=True)

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
        self.W1 = np.random.rand(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1  = np.zeros(hidden_size)

        # initialize weights and biases for hidden and output layer
        self.W2 = np.random.rand(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.loss_hist = []

        # test
        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)
