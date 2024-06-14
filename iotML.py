import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

client_1_data = pd.read_csv("/Users/mansahaj/cybersecurity_nrc/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
client_2_data = pd.read_csv("/Users/mansahaj/cybersecurity_nrc/MachineLearningCVE/Tuesday-workingHours.pcap_ISCX.csv")
client_3_data = pd.read_csv("/Users/mansahaj/cybersecurity_nrc/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")
df = pd.concat([client_1_data, client_2_data, client_3_data], ignore_index=True)

df[' Label'] = df[' Label'].apply(lambda x: 1 if 'BENIGN' in x else 0)

df = df.drop_duplicates(keep='first')

one_value = df.columns[df.nunique() == 1]
df2 = df.drop(columns = one_value, axis=1)

df2['Flow Bytes/s'] = df2['Flow Bytes/s'].fillna(df2['Flow Bytes/s'].mean())
df2.rename(columns=lambda x: x.lstrip(), inplace=True)
df2 = df2.drop(['Flow Packets/s', 'Flow Bytes/s'], axis=1)

X = df2.drop('Label', axis=1)
y = df2['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_ann_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train ANN model
ann_model = create_ann_model(X_train.shape[1])
ann_model.fit(X_train, y_train, epochs=4, batch_size=32, validation_split=0.1)

# # Evaluate ANN
ann_accuracy = ann_model.evaluate(X_test, y_test)[1]*100
precision = precision_score(y_test, y_test)
recall = recall_score(y_test, y_test)
f1 = f1_score(y_test, y_test)

print(f'Accuracy: {ann_accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')



# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# Predict the test data
# y_pred = model.predict(X_test)

# # Calculate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')