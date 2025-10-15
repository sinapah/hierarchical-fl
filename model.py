import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from geopy.distance import geodesic

# --- CONFIG ---
CSV_PATH = "./data/chicago_taxi_trips.csv"
N_DERMS = 8            # number of neighborhood DERMS
MIN_ROWS_PER_TAXI = 5  # drop taxis with too little data for local training
# ----------------

# 1) Load and clean
df = pd.read_csv(CSV_PATH)
# ensure pickup columns exist; adapt names if different
lat_col = "pickup_centroid_latitude"
lon_col = "pickup_centroid_longitude"
tid_col = "taxi_id"
time_col = "trip_start_timestamp"  # replace with your timestamp column

# drop missing coords (you decided earlier to remove these)
df = df.dropna(subset=[lat_col, lon_col])

# (optional) parse timestamp if present
if time_col in df.columns:
    df[time_col] = pd.to_datetime(df[time_col])

# 2) Build DERMS via clustering on pickup coords
coords = df[[lat_col, lon_col]].values
kmeans = KMeans(n_clusters=N_DERMS, random_state=42)
df["derms_id"] = kmeans.fit_predict(coords)

# Save DERMS centroids for inspection / mapping
derms_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=[lat_col, lon_col])
derms_centroids["derms_id"] = derms_centroids.index

# 3) Example: Define a simple per-trip target and features
# Here we predict trip duration (seconds) from trip_miles and hour-of-day
# adapt column names to your dataset
if "trip_seconds" not in df.columns and "trip_miles" in df.columns:
    # if you have trip_start and trip_end, compute trip_seconds; otherwise skip
    pass

# Simple feature engineering (safe-if columns exist)
features = []
if "trip_miles" in df.columns:
    features.append("trip_miles")
if time_col in df.columns:
    df["hour"] = df[time_col].dt.hour
    features.append("hour")

target = "trip_seconds"  # change if you prefer a different target

# Drop taxis with too few records to train a meaningful local model
taxi_counts = df[tid_col].value_counts()
valid_taxis = taxi_counts[taxi_counts >= MIN_ROWS_PER_TAXI].index
df = df[df[tid_col].isin(valid_taxis)].copy()

# 4) Build taxi-level local models **per DERMS**.
# Note: a taxi may appear under multiple DERMS over time; training is done
# on the subset of that taxi's rows that fall in each DERMS (i.e., what that DERMS sees).
local_models = {}  # (derms_id, taxi_id) -> model params

for derms_id, derms_df in df.groupby("derms_id"):
    for taxi_id, taxi_df in derms_df.groupby(tid_col):
        taxi_df = taxi_df.dropna(subset=features + [target])
        if len(taxi_df) < MIN_ROWS_PER_TAXI:
            continue
        if target not in taxi_df.columns or not features:
            continue
        X = taxi_df[features].values
        y = taxi_df[target].values
        # train simple linear regression as "local model"
        model = LinearRegression().fit(X, y)
        # store coefficients and intercept (easier to aggregate)
        local_models[(derms_id, taxi_id)] = {
            "coef": model.coef_.copy(),
            "intercept": float(model.intercept_),
            "n_samples": len(taxi_df)
        }

# 5) DERMS-level aggregation: average taxi models weighted by number of local samples
derms_aggregates = {}
for derms_id, group in df.groupby("derms_id"):
    # collect all taxi models in this derms
    models = [local_models[k] for k in local_models.keys() if k[0] == derms_id]
    if not models:
        continue
    # weighted average by n_samples
    total_n = sum(m["n_samples"] for m in models)
    avg_coef = sum(m["coef"] * m["n_samples"] for m in models) / total_n
    avg_intercept = sum(m["intercept"] * m["n_samples"] for m in models) / total_n
    derms_aggregates[derms_id] = {"coef": avg_coef, "intercept": avg_intercept, "n_samples": total_n}

# 6) Global aggregation: average DERMS models (weighted by DERMS n_samples)
if derms_aggregates:
    tot = sum(v["n_samples"] for v in derms_aggregates.values())
    global_coef = sum(v["coef"] * v["n_samples"] for v in derms_aggregates.values()) / tot
    global_intercept = sum(v["intercept"] * v["n_samples"] for v in derms_aggregates.values()) / tot
else:
    global_coef = None
    global_intercept = None

# 7) Quick evaluation of the global model (if features/target present)
if global_coef is not None:
    X_test = df[features].values
    y_test = df[target].values
    y_pred = X_test.dot(global_coef) + global_intercept
    mask = (
    np.isfinite(y_test) &
    np.isfinite(y_pred)
)

    if mask.sum() == 0:
        print("No valid samples for evaluation.")
    else:
        r2 = r2_score(y_test[mask], y_pred[mask])
        print(f"Global model R²: {r2:.4f} (using {mask.sum()} valid samples)")
else:
    print("No global model produced (missing features/target or no local models).")

# 8) Example: for a given new trip (lat,lng), determine which DERMS it should report to
def nearest_derms(lat, lon, centroids=derms_centroids):
    # compute Euclidean on lat/lon (ok for small areas) or use geodesic for more accuracy
    dists = np.sqrt((centroids[lat_col] - lat)**2 + (centroids[lon_col] - lon)**2)
    return int(centroids.iloc[dists.argmin()]["derms_id"])

sample_lat, sample_lon = df.iloc[0][lat_col], df.iloc[0][lon_col]
print("sample row reports to DERMS:", nearest_derms(sample_lat, sample_lon))

