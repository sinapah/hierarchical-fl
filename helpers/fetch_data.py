import requests
import pandas as pd

url = "https://data.cityofchicago.org/resource/wrvz-psew.json"

limit = 100000
offset = 0
total_rows = 200000

all_data = []

while offset < total_rows:
    params = {
        "$limit": limit,
        "$offset": offset
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        offset += limit

        print(f"Fetched {len(data)} records. Total records so far: {len(all_data)}")

    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        break

df = pd.DataFrame(all_data)
df.to_csv("chicago_taxi_trips_new.csv", index=False)
print(f"Data has been saved to chicago_taxi_trips.csv with {len(df)} records.")

