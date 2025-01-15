import os
import tarfile
import subprocess

# Define the library name and destination folder
library_name = 'smogn'
destination_folder = '/path/to/destination_folder'

# Download the source distribution
subprocess.run(['pip', 'download', '--no-binary', ':all:', '--no-deps', library_name])

# Find the downloaded tar.gz file
tar_gz_file = next(f for f in os.listdir() if f.startswith(library_name) and f.endswith('.tar.gz'))

# Extract the contents
with tarfile.open(tar_gz_file, 'r:gz') as tar:
    tar.extractall(path=destination_folder)

# Clean up the downloaded tar.gz file
os.remove(tar_gz_file)