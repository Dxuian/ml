import pandas as pd

# Load the dataset
# Replace 'your_dataset.csv' with the actual path to your dataset file
df = pd.read_csv('edf.csv')

# Calculate the percentage of loan_status values that are less than 0.5
percentage = (df['loan_status'] < 0.5).mean() * 100

# Print the result
print(f"Percentage of loan_status values less than 0.5: {percentage:.2f}%")