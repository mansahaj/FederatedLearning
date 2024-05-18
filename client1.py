# importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import flwr as fl
import tensorflow as tf
import io

def load_data(filepath):
    # data pre processing
    df = pd.read_csv(filepath)
    df[' Label'] = df[' Label'].apply(lambda x: 1 if 'BENIGN' in x else 0)

    df = df.drop_duplicates(keep='first')

    one_value = df.columns[df.nunique() == 1]
    df2 = df.drop(columns = one_value, axis=1)

    df2['Flow Bytes/s'] = df2['Flow Bytes/s'].fillna(df2['Flow Bytes/s'].mean())
    df2 = df2.drop([' Flow Packets/s', 'Flow Bytes/s'], axis=1)

    X = df2.drop(' Label', axis=1)
    y = df2[' Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_ann_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class KerasFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        # Return model weights as a list of NumPy arrays
        return [weight for weight in self.model.get_weights()]
    
    def set_parameters(self, parameters):
        # Set model weights from the list of NumPy arrays
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.get_parameters(config), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

x_train, x_test, y_train, y_test = load_data("/Users/mansahaj/cybersecurity_nrc/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
input_shape = (x_train.shape[1],)  # Assuming x_train is a DataFrame or a 2D NumPy array

model = create_ann_model(input_shape)  # or create_cnn_model(input_shape)
client = KerasFlowerClient(model, x_train, y_train, x_test, y_test)
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client.to_client())
