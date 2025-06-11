from dotenv import load_dotenv
import os

# Update this to the local synced folder path
folder_path = os.getenv("FOLDER")

# List all files
for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    if os.path.isfile(full_path):
        print(f"Found file: {filename}")
        # Add your logic here (e.g., read file, extract text)
