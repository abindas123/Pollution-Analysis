import pandas as pd

file_path = 'C:\\Users\\Abindas\\Downloads\\Assignment 2\\AirQuality.csv'

# Try different delimiters and encodings
try:
    df = pd.read_csv(file_path, encoding='utf-8', delimiter=';')  # Change delimiter if needed
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')

# ✅ Print column names BEFORE renaming
print("Columns before renaming:", df.columns.tolist())
