import pandas as pd

# Load the DataFrame from the CSV file
df = pd.read_csv("okaya.csv")

# Calculate the current standard deviation of the 'loan_status' column
current_std_dev = df['loan_status'].std()

# Define the target standard deviation
target_std_dev = current_std_dev * 0.98

# Scale the 'loan_status' values to achieve the target standard deviation
mean = df['loan_status'].mean()
df['loan_status'] = mean + (df['loan_status'] - mean) * (target_std_dev / current_std_dev)
# Display the modified DataFrame

# Optionally, save the modified DataFrame to a new CSV file
df.to_csv("okaya_modified.csv",index=False)
