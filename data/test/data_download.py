from sklearn.datasets import fetch_openml

# Replace data_id with any OpenML ID (e.g., 1461 for bank-marketing)
data = fetch_openml(data_id=31, as_frame=True, parser="auto")
df = data.frame

# Save directly as a CSV file
df.to_csv("credit_g.csv", index=False)
df.to_csv("../credit_g.csv", index=False)
print("Saved as credit_g.csv successfully!")