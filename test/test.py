import joblib
import numpy as np
from tensorflow import keras
# from sklearn.externals import joblib

# Generate random data
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

# Define a sequential model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=20),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=2000,
          batch_size=128)
joblib.dump(model, 'model2.joblib')