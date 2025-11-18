import pandas as pd
import random
import numpy as np
from faker import Faker

random.seed(42)
np.random.seed(42)
fake = Faker("en_US")

# NUM_CUSTOMERS = 2000

region_data = [
    {"state": "Minnesota", "city": "Minneapolis", "zipcodes": ["55414", "55401", "55419"]},
    {"state": "New York", "city": "New York", "zipcodes": ["10001", "10002", "10003"]},
    {"state": "Illinois", "city": "Chicago", "zipcodes": ["60601", "60602", "60603"]},
    {"state": "Texas", "city": "Dallas", "zipcodes": ["75201", "75202", "75203"]},
    {"state": "California", "city": "San Jose", "zipcodes": ["95101", "95110", "95112"]},
    {"state": "Florida", "city": "Miami", "zipcodes": ["33101", "33125", "33130"]},
    {"state": "Washington", "city": "Seattle", "zipcodes": ["98101", "98102", "98103"]},
]

occupations = ["teacher", "engineer", "doctor", "driver", "freelancer", "manager", "retired", "salesperson", "student"]
education_levels = ["high_school", "bachelor", "master", "phd"]
employment_statuses = ["employed", "self-employed", "unemployed", "retired"]

occupations_by_age = {
    "young": ["student", "intern", "assistant", "junior developer", "sales trainee"],
    "adult": ["teacher", "engineer", "doctor", "manager", "freelancer", "salesperson", "technician"],
    "senior": ["retired", "consultant", "advisor", "shop owner"]
}

education_by_age = {
    "young": ["high_school", "bachelor"],
    "adult": ["bachelor", "master"],
    "senior": ["bachelor", "master", "phd"]
}

"""
def generate_customer_data():  
    
    name = fake.name()
    # email = fake.email()
    phone = fake.phone_number()

    region = random.choice(region_data)
    state, city = region["state"], region["city"]
    zipcode = random.choice(region["zipcodes"])

    # Core demographics
    age = random.randint(20, 75)
    gender = random.choice(["male", "female", "nonbinary"])
    
    if age < 25:
        marital_status = random.choices(["single", "married"], weights=[0.9, 0.1])[0]
    elif 25 <= age < 40:
        marital_status = random.choices(["single", "married", "divorced"], weights=[0.3, 0.6, 0.1])[0]
    elif 40 <= age < 60:
        marital_status = random.choices(["married", "divorced", "widowed"], weights=[0.6, 0.3, 0.1])[0]
    else:  # 60+
        marital_status = random.choices(["married", "divorced", "widowed"], weights=[0.5, 0.2, 0.3])[0]
    
    dependents = 0 if marital_status == "single" else random.randint(1, 3)
   
    if age < 25:
        occupation = random.choice(occupations_by_age["young"])
        education = random.choice(education_by_age["young"])
        employment_status = "student" if occupation == "student" else "employed"
    elif 25 <= age < 60:
        occupation = random.choice(occupations_by_age["adult"])
        education = random.choice(education_by_age["adult"])
        employment_status = random.choice(["employed", "self-employed"])
    else:
        occupation = random.choice(occupations_by_age["senior"])
        education = random.choice(education_by_age["senior"])
        employment_status = "retired"

    if employment_status == "employed":
        annual_income = random.randint(40000, 120000)
    elif employment_status == "self-employed":
        annual_income = random.randint(30000, 90000)
    elif employment_status == "retired":
        annual_income = random.randint(20000, 60000)
    else:
        annual_income = random.randint(15000, 40000)


    credit_score = int(np.clip(np.random.normal(700, 60), 550, 850))
    avg_monthly_expense = round(annual_income / random.uniform(10, 15), 2)
    existing_loans = random.choice(["none", "home", "car", "personal", "multiple"])


    vehicle_owner = random.random() < 0.7
    vehicle_type = random.choice(["car", "bike", "none"]) if vehicle_owner else "none"
    
    # coverage_pref = random.choice(coverage_types)
    budget_per_month = random.randint(100, 400)

    tenure = round(random.uniform(0.5, 10), 1)
     


    return {
        "customer_id": f"CUST_{random.randint(100000, 999999)}", # TODO: Change customer id
        "name": name,
        # "email": email,
        "phone": phone,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "education_level": education,
        "occupation": occupation,
        "employment_status": employment_status,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "avg_monthly_expense": avg_monthly_expense,
        "vehicle_owner": vehicle_owner,
        "vehicle_type": vehicle_type,
        # "coverage_preference": coverage_pref, # TODO: change to multiple preferences
        "budget_per_month": budget_per_month,
        # "customer_tenure_years": tenure
    }
    
"""

def generate_customer_data(existing_ids=set()):
    name = fake.name()
    phone = fake.phone_number()

    # Region
    region = random.choice(region_data)
    state, city = region["state"], region["city"]
    zipcode = random.choice(region["zipcodes"])

    # Demographics
    age = random.randint(20, 75)
    gender = random.choice(["male", "female", "nonbinary"])
    
    if age < 25:
        marital_status = random.choices(["single", "married"], weights=[0.9, 0.1])[0]
    elif 25 <= age < 40:
        marital_status = random.choices(["single", "married", "divorced"], weights=[0.3, 0.6, 0.1])[0]
    elif 40 <= age < 60:
        marital_status = random.choices(["married", "divorced", "widowed"], weights=[0.6, 0.3, 0.1])[0]
    else:
        marital_status = random.choices(["married", "divorced", "widowed"], weights=[0.5, 0.2, 0.3])[0]

    dependents = 0 if marital_status == "single" else random.randint(1, 3)
   
    # Occupation & Education
    if age < 25:
        occupation = random.choice(occupations_by_age["young"])
        education = random.choice(education_by_age["young"])
        employment_status = "student" if occupation == "student" else "employed"
    elif 25 <= age < 60:
        occupation = random.choice(occupations_by_age["adult"])
        education = random.choice(education_by_age["adult"])
        employment_status = random.choice(["employed", "self-employed"])
    else:
        occupation = random.choice(occupations_by_age["senior"])
        education = random.choice(education_by_age["senior"])
        employment_status = "retired"

    # Income
    if employment_status == "employed":
        annual_income = random.randint(40000, 120000)
    elif employment_status == "self-employed":
        annual_income = random.randint(30000, 90000)
    elif employment_status == "retired":
        annual_income = random.randint(20000, 60000)
    else:
        annual_income = random.randint(15000, 40000)

    # Financial & lifestyle
    credit_score = int(np.clip(np.random.normal(700, 60), 550, 850))
    avg_monthly_expense = round(annual_income / random.uniform(10, 15), 2)
    existing_loans = random.choice(["none", "home", "car", "personal", "multiple"])
    vehicle_owner = random.random() < 0.7
    vehicle_type = random.choice(["car", "bike", "none"]) if vehicle_owner else "none"
    budget_per_month = random.randint(100, 400)
    tenure = round(random.uniform(0.5, 10), 1)

    # Coverage preferences (1–2 types)
    coverage_types = ["health", "life", "auto", "home", "travel"]
    coverage_preferences = random.sample(coverage_types, k=random.randint(1,2))

    # Risk tolerance
    risk_tolerance = random.choice(["Low", "Medium", "High"])

    possible_tags = ["family", "maternity", "travel", "vehicle", "retirement", "education"]
    interest_tags = []

    if dependents > 0:
        interest_tags.append("family")
        if random.random() < 0.5:
            interest_tags.append("education")
        if random.random() < 0.3:
            interest_tags.append("maternity")

    if vehicle_owner:
        interest_tags.append("vehicle")

    if age >= 60:
        interest_tags.append("retirement")
    if age < 25:
        interest_tags.append("education")

    if annual_income > 50000 and budget_per_month > 150:
        if random.random() < 0.5:
            interest_tags.append("travel")
            
    interest_tags = list(set(interest_tags))


    while True:
        customer_id = f"CUST_{random.randint(100000, 999999)}"
        if customer_id not in existing_ids:
            existing_ids.add(customer_id)
            break

    return {
        "customer_id": customer_id,
        "name": name,
        "phone": phone,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "education_level": education,
        "occupation": occupation,
        "employment_status": employment_status,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "avg_monthly_expense": avg_monthly_expense,
        "vehicle_owner": vehicle_owner,
        "vehicle_type": vehicle_type,
        "budget_per_month": budget_per_month,
        "customer_tenure_years": tenure,
        "coverage_preferences": coverage_preferences,
        "risk_tolerance": risk_tolerance,
        "interest_tags": interest_tags
    }
  
    
