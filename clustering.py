import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import hdbscan

df = pd.read_csv("customers.csv")
df_original = df.copy()

drop_cols = [
    "customer_id", "name", "phone", "city", "state", "zipcode", "interest_tags"
]
df = df.drop(columns=drop_cols, errors='ignore')

# 3. Encode Categorical Columns
categorical_cols = [
    "gender", "marital_status", "education_level",
    "occupation", "employment_status", "vehicle_owner",
    "vehicle_type", "coverage_preferences", "risk_tolerance"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 4. Fill Missing Values (safe version)
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 5. Feature Selection
features = [
    "age", "gender", "marital_status", "dependents", "education_level",
    "occupation", "employment_status", "annual_income", "credit_score",
    "existing_loans", "avg_monthly_expense", "vehicle_owner",
    "budget_per_month", "customer_tenure_years", "coverage_preferences",
    "risk_tolerance"
]
features = [f for f in features if f in df.columns]


X = df[features].apply(pd.to_numeric, errors='coerce')

for col in X.columns:
    if X[col].isna().all():
        print(f"Dropping {col} (entirely NaN after encoding)")
        X = X.drop(columns=[col])
    else:
        X[col] = X[col].fillna(X[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

best_score = -1
best_k = 0
for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_umap)
    score = silhouette_score(X_umap, labels)
    if score > best_score:
        best_score = score
        best_k = k

print(f"UMAP + KMeans → Best k: {best_k}, Silhouette Score: {best_score:.3f}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Segment_ID"] = kmeans.fit_predict(X_umap)

umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
umap_df["Segment"] = df["Segment_ID"]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=umap_df,
    x="UMAP1", y="UMAP2",
    hue="Segment",
    palette="Set2",
    s=70,
    edgecolor='black',
    alpha=0.8
)
plt.title("Customer Segments via UMAP + KMeans", fontsize=14)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


print()
print(df.head())
print()
df_original["Segment_ID"] = df["Segment_ID"]
df_original.to_csv("customers_clustered.csv", index=False)
# df.to_csv("customers_clustered.csv", index=False)


numeric_features = df[features].select_dtypes(include=[np.number]).columns

summary = df.groupby("Segment_ID")[numeric_features].mean().round(2)
print("\nCluster Summary (Numeric Features Only):")
print(summary)
summary.to_csv("cluster_summary_numeric.csv")

cat_features = [col for col in features if col not in numeric_features]
if cat_features:
    cat_summary = df.groupby("Segment_ID")[cat_features].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    print("\nMost Common Category per Cluster:")
    print(cat_summary)


