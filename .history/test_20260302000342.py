import pandas as pd
import os

basePath = os.getcwd()
print(basePath)
folder_path = os.join(basePath, "Data/TNG/Dev")
output_file = "combined.csv"

first_file = True

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)

        df = pd.read_csv(file_path)

        df.to_csv(
            output_file,
            mode="w" if first_file else "a",
            header=first_file,
            index=False
        )

        first_file = False

print("All CSV files combined.")