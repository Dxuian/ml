import pandas as pd

# Sample data for df1 with columns a, b, c, target
data1 = {
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'target': [10, 11, 12]
}
df1 = pd.DataFrame(data1)

# Sample data for df2 with columns a, b, c
data2 = {
    'a': [13, 14, 15],
    'b': [16, 17, 18],
    'c': [19, 20, 21]
}
df2 = pd.DataFrame(data2)

# Select only the columns a, b, and c from df1
df1_selected = df1[['a', 'b', 'c']]

# Concatenate df1_selected and df2
result = pd.concat([df1_selected, df2], ignore_index=True)

# Display the result
print("Resulting DataFrame:")
print(result)