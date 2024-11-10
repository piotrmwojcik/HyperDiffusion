# Define the path to the file containing the data
file_path = './all_vox256_img_train.txt'

# Initialize a set to store unique prefixes
unique_prefixes = set()

# Open and read the file
with open(file_path, 'r') as file:
    for line in file:
        # Split each line by '/' and extract the prefix (first part)
        prefix = line.split('/')[0]
        unique_prefixes.add(prefix)  # Add the prefix to the set

# Convert the set to a sorted list (optional)
unique_prefixes = sorted(unique_prefixes)

# Display the unique prefixes
print("Unique prefixes found:")
print(len(unique_prefixes))
for prefix in unique_prefixes:
    print(prefix)