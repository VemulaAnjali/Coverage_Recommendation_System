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


def get_policy_details(policy_type, policies, customer_ids=None):
    if customer_ids is not None:
        relevant = policies[(policies['policy_type'] == policy_type) &
                            (policies['customer_id'].isin(customer_ids))]
    else:
        relevant = policies[policies['policy_type'] == policy_type]
    
    if len(relevant) == 0:
        return {"policy_type": policy_type, "coverage_amount": "nan", "premium": "nan"}
    
    coverage_amount = relevant['coverage_amount'].mean()
    premium = relevant['premium_amount'].mean()
    
    return {"policy_type": policy_type,
            "coverage_amount": coverage_amount,
            "premium": premium}


def recommend_agents_for_policy(customer_state, policy_type, agents_df, top_n=2):

    filtered_agents = agents_df[
        agents_df['region'].str.contains(str(customer_state), case=False, na=False)
    ].copy()


    if len(filtered_agents) < top_n:
        filtered_agents = agents_df[
            agents_df['specialization'].str.contains(str(policy_type), case=False, na=False)
        ].copy()

    if len(filtered_agents) < top_n:
        filtered_agents = agents_df.copy()

    filtered_agents['score'] = filtered_agents['success_rate']*0.6 + filtered_agents['average_rating']*0.4
    filtered_agents = filtered_agents.sort_values(by='score', ascending=False)

    return filtered_agents.head(top_n)[['agent_id','agent_name','region','specialization',
                                        'experience_years','success_rate','average_rating',
                                        'active_clients','languages','contact_number','email']].to_dict(orient='records')


def recommend_policies_with_agents(new_customer, top_n=2, n_clusters=4, agents_df=None):
    
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

    policy_vectors = {}
    for pid, group in policies.groupby('policy_type'):
        indices = clustered_customers.index[clustered_customers['customer_id'].isin(group['customer_id'])]
        if len(indices) > 0:
            policy_vectors[pid] = customer_features[indices].mean(axis=0)


    new_customer_num = scaler.transform([[new_customer[col] for col in numeric_cols]])
    new_customer_cat = ohe.transform([[new_customer[col] for col in categorical_cols]])
    new_customer_vector = np.hstack([new_customer_num, new_customer_cat])


    similarities = {pid: cosine_similarity(new_customer_vector, vec.reshape(1, -1))[0][0]
                    for pid, vec in policy_vectors.items()}
    top_cosine_policies = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    cosine_recommendations = []
    for pid in top_cosine_policies:
        policy_details = get_policy_details(pid, policies)
        if agents_df is not None:
            policy_details['recommended_agents'] = recommend_agents_for_policy(
                new_customer.get('city','nan'), pid, agents_df, top_n=2
            )
        cosine_recommendations.append(policy_details)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clustered_customers['Segment_ID'] = kmeans.fit_predict(customer_features)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(customer_features)
    _, idx = knn.kneighbors(new_customer_vector)
    nearest_customer = clustered_customers.iloc[idx[0][0]]
    nearest_cluster = nearest_customer['Segment_ID']

    cluster_customers = clustered_customers[clustered_customers['Segment_ID'] == nearest_cluster]
    cluster_policies = policies[policies['customer_id'].isin(cluster_customers['customer_id'])]
    cluster_policy_counts = cluster_policies['policy_type'].value_counts()
    top_cluster_policies = cluster_policy_counts.head(top_n).index.tolist()
    
    cluster_recommendations = []
    for pid in top_cluster_policies:
        policy_details = get_policy_details(pid, policies, cluster_customers['customer_id'])
        if agents_df is not None:
            policy_details['recommended_agents'] = recommend_agents_for_policy(
                new_customer.get('city','nan'), pid, agents_df, top_n=2
            )
        cluster_recommendations.append(policy_details)

    return {
        "cosine_similarity_recommendations": cosine_recommendations,
        "cluster_based_recommendations": cluster_recommendations
    }


agents_df = pd.read_csv("agents.csv") 

new_customer = {'name': 'Sarah', 'phone': 'nan', 'city': 'nan', 'state': 'nan', 'zipcode': 'nan', 'age': 'nan', 'gender': 'nan', 'marital_status': 'nan', 'dependents': 'nan', 'education_level': 'nan', 'occupation': 'nan', 'employment_status': 'Full-time', 'annual_income': 'nan', 'credit_score': 'nan', 'existing_loans': 'nan', 'avg_monthly_expense': 'nan', 'vehicle_owner': 'Yes', 'vehicle_type': 'nan', 'budget_per_month': 'nan', 'customer_tenure_years': 'nan', 'risk_tolerance': 'nan'}


new_customer = normalize_customer_input(new_customer)
recommendations = recommend_policies_with_agents(new_customer, top_n=3, agents_df=agents_df)
print(recommendations)
