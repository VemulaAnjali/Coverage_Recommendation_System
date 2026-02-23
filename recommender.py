# recommender.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

# Columns used in the model
NUMERIC_COLS = [
    "age",
    "annual_income",
    "credit_score",
    "avg_monthly_expense",
    "budget_per_month",
    "customer_tenure_years",
]

CATEGORICAL_COLS = [
    "gender",
    "marital_status",
    "education_level",
    "occupation",
    "employment_status",
    "vehicle_owner",
    "vehicle_type",
    "risk_tolerance",
]


def recommend_policies(new_customer, top_n=2, n_clusters=4):
    # Load data
    clustered_customers = pd.read_csv("customers_clustered.csv")
    policies = pd.read_csv("policies.csv")

    # ---------- Feature preparation ----------
    # Numeric
    scaler = StandardScaler()
    customer_num = scaler.fit_transform(clustered_customers[NUMERIC_COLS])

    # Categorical
    # handle_unknown="ignore" is IMPORTANT to avoid errors
    # when new_customer has categories not seen in training data
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    customer_cat = ohe.fit_transform(clustered_customers[CATEGORICAL_COLS])

    # Combined feature matrix
    customer_features = np.hstack([customer_num, customer_cat])
    clustered_customers["feature_vector"] = list(customer_features)

    # ---------- Build policy vectors (average of customers per policy_type) ----------
    policy_vectors = {}
    for pid, group in policies.groupby("policy_type"):
        indices = clustered_customers.index[
            clustered_customers["customer_id"].isin(group["customer_id"])
        ]
        if len(indices) > 0:
            policy_vectors[pid] = customer_features[indices].mean(axis=0)

    # ---------- New customer vector ----------
    new_customer_num = scaler.transform([[new_customer[col] for col in NUMERIC_COLS]])
    new_customer_cat = ohe.transform([[new_customer[col] for col in CATEGORICAL_COLS]])
    new_customer_vector = np.hstack([new_customer_num, new_customer_cat])

    # ---------- Cosine similarity recommendations ----------
    similarities = {
        pid: cosine_similarity(new_customer_vector, vec.reshape(1, -1))[0][0]
        for pid, vec in policy_vectors.items()
    }
    cosine_recommendations = sorted(similarities, key=similarities.get, reverse=True)[
        :top_n
    ]

    # ---------- Cluster + KNN based recommendations ----------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clustered_customers["Segment_ID"] = kmeans.fit_predict(customer_features)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(customer_features)
    _, idx = knn.kneighbors(new_customer_vector)
    nearest_cluster = clustered_customers.iloc[idx[0][0]]["Segment_ID"]

    cluster_customers = clustered_customers[
        clustered_customers["Segment_ID"] == nearest_cluster
    ]
    cluster_policies = policies[
        policies["customer_id"].isin(cluster_customers["customer_id"])
    ]
    cluster_recommendations = (
        cluster_policies["policy_type"]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    return {
        "cosine_similarity_recommendations": cosine_recommendations,
        "cluster_based_recommendations": cluster_recommendations,
    }
