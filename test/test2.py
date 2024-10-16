import numpy as np
import time

# Generate two large random lists
list1 = np.random.randint(1, 100, size=100000000).tolist()
list2 = np.random.randint(1, 100, size=100000000).tolist()

# Multiply using a loop
start_time = time.time()
result_loop = [a*b for a, b in zip(list1, list2)]
end_time = time.time()
print(f"Time taken using loop: {end_time - start_time} seconds")

# Convert lists to numpy arrays
np_array1 = np.array(list1)
np_array2 = np.array(list2)

# Multiply using np.dot
start_time = time.time()
result_np_dot = np.dot(np_array1, np_array2)
end_time = time.time()
print(f"Time taken using np.dot: {end_time - start_time} seconds")