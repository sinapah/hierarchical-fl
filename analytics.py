import pandas as pd

df = pd.read_csv("chicago_taxi_trips.csv")

if "taxi_id" not in df.columns:
    raise ValueError("The dataset does not have a 'taxi_id' column.")

id_counts = df["taxi_id"].value_counts()

repeated_ids = id_counts[id_counts > 1]

if repeated_ids.empty:
    print("Each taxi_id appears only once in the dataset.")
else:
    print(f"{len(repeated_ids)} taxi_id(s) that appear more than once.")
    print("\nExample of repeated taxi_ids and their counts:")
    print(repeated_ids.head(10))

id_counts.to_csv("taxi_id_counts.csv", header=["count"])
print("\nSaved 'taxi_id_counts.csv' for detailed inspection.")

