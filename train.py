import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

DATA_DIR = 'asl_dataset'
data = []
labels = []

for label in os.listdir(DATA_DIR):
    for file in os.listdir(os.path.join(DATA_DIR, label)):
        csv_path = os.path.join(DATA_DIR, label, file)
        features = np.loadtxt(csv_path, delimiter=',')
        data.append(features)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

model.save('asl_model.h5')
print("✅ Model trained and saved as asl_model.h5")
