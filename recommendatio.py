import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

def normalize_customer_input(customer):
    numeric_cols = [
        'age','annual_income','credit_score',
        'avg_monthly_expense','budget_per_month','customer_tenure_years'
    ]
    
    categorical_cols = [
        'gender','marital_status','education_level','occupation',
        'employment_status','vehicle_owner','vehicle_type','risk_tolerance'
    ]

    cleaned = {}

    for col in numeric_cols:
        val = customer.get(col, None)
        if val is None or val == "" or str(val).lower() in ["nan", "none", "null"]:
            cleaned[col] = 0
        else:
            try:
                cleaned[col] = float(val)
            except:
                cleaned[col] = 0

    for col in categorical_cols:
        val = customer.get(col, None)
        if val is None or val == "" or str(val).lower() in ["nan", "none", "null"]:
            cleaned[col] = "nan"
        else:
            cleaned[col] = str(val).strip()

    for key, value in customer.items():
        if key not in numeric_cols and key not in categorical_cols:
            cleaned[key] = value

    return cleaned


def recommend_policies(new_customer, top_n=2, n_clusters=4):
    
    clustered_customers = pd.read_csv("customers_clustered.csv")
    policies = pd.read_csv("policies.csv")

    numeric_cols = ['age','annual_income','credit_score','avg_monthly_expense','budget_per_month','customer_tenure_years']
    categorical_cols = ['gender','marital_status','education_level','occupation','employment_status','vehicle_owner','vehicle_type','risk_tolerance']

    scaler = StandardScaler()
    customer_num = scaler.fit_transform(clustered_customers[numeric_cols])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    customer_cat = ohe.fit_transform(clustered_customers[categorical_cols])

    customer_features = np.hstack([customer_num, customer_cat])
    clustered_customers['feature_vector'] = list(customer_features)

    # policy_vectors = {}
    # for pid, group in policies.groupby('policy_id'):
    #     indices = clustered_customers[clustered_customers['customer_id'].isin(group['customer_id'])].index
    #     policy_vectors[pid] = customer_features[indices].mean(axis=0)
        
    policy_vectors = {}
    for pid, group in policies.groupby('policy_type'):
        indices = clustered_customers.index[clustered_customers['customer_id'].isin(group['customer_id'])]
        if len(indices) > 0:
            policy_vectors[pid] = customer_features[indices].mean(axis=0)


    new_customer_num = scaler.transform([ [new_customer[col] for col in numeric_cols] ])
    new_customer_cat = ohe.transform([ [new_customer[col] for col in categorical_cols] ])
    new_customer_vector = np.hstack([new_customer_num, new_customer_cat])
    
    similarities = {pid: cosine_similarity(new_customer_vector, vec.reshape(1, -1))[0][0]
                for pid, vec in policy_vectors.items()}

    cosine_recommendations = sorted(similarities, key=similarities.get, reverse=True)[:top_n]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clustered_customers['Segment_ID'] = kmeans.fit_predict(customer_features)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(customer_features)
    _, idx = knn.kneighbors(new_customer_vector)
    nearest_cluster = clustered_customers.iloc[idx[0][0]]['Segment_ID']

    cluster_customers = clustered_customers[clustered_customers['Segment_ID'] == nearest_cluster]
    cluster_policies = policies[policies['customer_id'].isin(cluster_customers['customer_id'])]
    cluster_recommendations = cluster_policies['policy_type'].value_counts().head(top_n).index.tolist()

    return {
        "cosine_similarity_recommendations": cosine_recommendations,
        "cluster_based_recommendations": cluster_recommendations
    }

# new_customer = {
#     "customer_id": "CUST_999999",
#     "name": "John Doe",
#     "phone": "555-123-4567",
#     "city": "San Jose",
#     "state": "California",
#     "zipcode": "95112",
#     "age": 30,
#     "gender": "male",
#     "marital_status": "single",
#     "dependents": 0,
#     "education_level": "bachelor",
#     "occupation": "engineer",
#     "employment_status": "employed",
#     "annual_income": 60000,
#     "credit_score": 700,
#     "existing_loans": "none",
#     "avg_monthly_expense": 1500,
#     "vehicle_owner": True,
#     "vehicle_type": "car",
#     "budget_per_month": 300,
#     "customer_tenure_years": 2,
#     "risk_tolerance": "Medium"
# }

new_customer = {'customer_id': 'nan', 'name': 'John Doe', 'phone': 'nan', 'city': 'San Jose', 'state': 'nan', 'zipcode': 'nan', 'age': 'nan', 'gender': 'nan', 'marital_status': 'nan', 'dependents': 'nan', 'education_level': 'nan', 'occupation': 'engineer', 'employment_status': 'Full Time', 'annual_income': 'nan', 'credit_score': 'nan', 'existing_loans': 'No', 'avg_monthly_expense': 'nan', 'vehicle_owner': 'Yes', 'vehicle_type': '2018 Honda Civic', 'budget_per_month': '300', 'customer_tenure_years': 'nan', 'risk_tolerance': 'nan'}

new_customer = normalize_customer_input(new_customer)
recommendations = recommend_policies(new_customer)
print(recommendations)
