import os
import subprocess

# Path to add
new_path = r'C:\Users\death\AppData\Roaming\Python\Python312\Scripts'

# Retrieve the current PATH environment variable
current_path = os.environ.get('PATH', '')

# Append the new path if it's not already in the PATH
if new_path not in current_path:
    updated_path = f"{current_path};{new_path}"
    
    # Update the PATH environment variable using setx command
    subprocess.run(['setx', 'PATH', updated_path], shell=True)
    print(f"Added {new_path} to PATH")
else:
    print(f"{new_path} is already in PATH")