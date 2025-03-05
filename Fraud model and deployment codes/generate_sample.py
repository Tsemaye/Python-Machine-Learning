import pandas as pd
import json

# Define the path to your dataset
dataset_path = "/Users/charisoneyemi/Downloads/611Assignment/Dataset/creditcard.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Select a sample row (without the 'Class' column)
sample_row = df.drop(columns=["Class"]).iloc[0].to_dict()

# Print the JSON (to use in Postman)
print(json.dumps({"data": sample_row}, indent=4))
