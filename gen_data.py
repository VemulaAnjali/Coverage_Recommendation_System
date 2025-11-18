import pandas as pd
import random
from faker import Faker
from gen_customer import generate_customer_data
from gen_policy import generate_policy
from gen_agent import generate_agents


NUM_CUSTOMERS = 5000

customers = [generate_customer_data() for _ in range(NUM_CUSTOMERS)]
customers_df = pd.DataFrame(customers)  
policies = []
for customer in customers:
    customer_policies = generate_policy(customer)
    policies.extend(customer_policies)
policies_df = pd.DataFrame(policies)
agents_df = generate_agents(50)

customers_df.to_csv("customers.csv", index=False)
policies_df.to_csv("policies.csv", index=False)
agents_df.to_csv("agents.csv", index=False)
print("Generated customers.csv, policies.csv, and agents.csv")
