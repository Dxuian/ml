 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn as sk
 
import tensorflow as tf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import itertools
import pandas as pd
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, r2_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the test.csv file
train = pd.read_csv('train.csv')
print(train.head())

# %%


# Plot histograms for numerical columns
numerical_columns = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
train[numerical_columns].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Plot count plots for categorical columns
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'loan_status']
fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # Adjusted to 3x2 grid
for ax, col in zip(axes.flatten(), categorical_columns):
    sns.countplot(data=train, x=col, ax=ax)
    ax.set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# %%


# Define the features and target
features = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
target = ['loan_status']  # Replace with the actual target column name

# Preprocess the data
X = train[features]
y = train[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the preprocessing for numerical and categorical features
numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


print(X_test.shape)
print(y_test.head())
print(y_test.shape)
print(y_train.head())
print(y_train.shape)
print(X_train.shape)
# Hardcoded array


# %%
# Print the shapes of the datasets
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Print the first few rows of y_test and y_train without indices
print('y_test values:')
print(y_test.values[:5])

print('y_train values:')
print(y_train.values[:5])

# %%
class CustomModel:
    def __init__(self, layer_sizes):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
        
        for units in reversed(sorted(layer_sizes)):
            self.model.add(tf.keras.layers.Dense(units, activation='relu'))
        
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    def fit(self, X_train, y_train, X_test, y_test, epochs=1,verbose=0,batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=verbose,batch_size=batch_size)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X_test):
        y_pred_prob = self.model.predict(X_test)
        return (y_pred_prob > 0.5).astype(int)
    
    def summary(self):
        self.model.summary()

# %%
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, r2_score
import gc
# Assuming CustomModel class and other necessary code is already defined

array = [256, 128, 64, 32, 16, 8, 4]
batch_sizes = [16, 32, 64]
epoch_values = [1, 5, 10]

# Generate the power set of the array
power_set = list(itertools.chain.from_iterable(itertools.combinations(array, r) for r in range(1, len(array) + 1)))

# Initialize variables to keep track of the best model
best_model = None
best_roc_auc = 0

# Loop to add layers, batch sizes, and epochs, and evaluate the model
for subset in power_set:
    for batch_size in batch_sizes:
        for epochs in epoch_values:
            custom_model = CustomModel(subset)
            
            # Train the model
            custom_model.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
            
            # Evaluate the model
            loss, accuracy = custom_model.evaluate(X_test, y_test)
            
            # Predict probabilities for the test set
            y_pred = custom_model.predict(X_test)
            
            # Calculate the ROC AUC score
            roc_auc = roc_auc_score(y_test, y_pred)
            
            # Calculate the R² score
            r2 = r2_score(y_test, y_pred)
            
            # Print the metrics
            print(f'Layers: {subset}, Batch Size: {batch_size}, Epochs: {epochs}')
            print(f'  Loss: {loss}')
            print(f'  Accuracy: {accuracy}')
            print(f'  ROC AUC: {roc_auc}')
            print(f'  R² score: {r2}')
            
            # Check if this model is the best so far
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = custom_model
                custom_model.model.save("bm.keras")
            gc.collect()

# Print the best model summary
print("Best Model Summary:")
best_model.summary()

 
 
