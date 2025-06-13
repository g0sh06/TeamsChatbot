import os
from dotenv import load_dotenv
import textract

# Load environment variables from .env file
load_dotenv()
folder = os.getenv("FOLDER")

# Check if FOLDER is set
if not folder:
    print("⚠️  FOLDER not set in .env")
    exit()

# Normalize the folder path
folder = os.path.normpath(folder)

# Go through all files in the folder
for file in os.listdir(folder):
    path = os.path.normpath(os.path.join(folder, file))
    
    if os.path.isfile(path) and not file.endswith(".ini"):
        print(f"\nReading: {file}")
        try:
            text = textract.process(path).decode("utf-8")
            print(text[:300])  # Show first 300 characters
        except Exception as e:
            print(f"Error reading {file}: {e}")
