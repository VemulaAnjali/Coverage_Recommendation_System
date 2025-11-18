import pandas as pd
import random
import numpy as np
from faker import Faker

random.seed(42)
np.random.seed(42)
fake = Faker("en_US")

policy_types = ["health", "auto", "life", "home", "travel"] # TODO: VERIFY AGAIN!!!!
# providers = ["Aetna", "Allianz", "StateFarm", "Progressive", "Liberty Mutual", "BlueCross", "MetLife", "TravelGuard"]

"""   
def generate_policy(customer):
    policies = []
    policy_types = ["health", "auto", "life", "home", "travel"]

    # Determine how many policies a customer has
    num_policies = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]
    chosen_types = random.sample(policy_types, num_policies)

    for ptype in chosen_types:
        # Skip incompatible policy types based on customer attributes
        if ptype == "auto" and not customer["vehicle_owner"]:
            continue
        if ptype == "home" and customer["existing_loans"] == "none":
            # optional logic: assume they don’t own a home if no loan
            continue  

        # Policy basics
        policy_id = f"POL_{random.randint(100000, 999999)}"
        provider = fake.company()
        premium = random.randint(500, 2500)
        coverage_amount = random.randint(10000, 500000)
        duration_years = random.randint(1, 10)
        active = random.random() < 0.85  # 85% active policies

        start_year = random.randint(2015, 2024)
        if active:
            end_year = start_year + duration_years
        else:
            end_year = start_year + random.randint(1, duration_years)

        # Risk-based premium adjustment factors
        risk_factor = 1.0

        # Income effect: lower income → higher risk premium
        if customer["annual_income"] < 40000:
            risk_factor += 0.1
        elif customer["annual_income"] > 100000:
            risk_factor -= 0.05

        # Credit score effect
        if customer["credit_score"] < 600:
            risk_factor += 0.2
        elif customer["credit_score"] > 750:
            risk_factor -= 0.1

        # Employment type: self-employed can be riskier for life/health
        if ptype in ["life", "health"] and customer["employment_status"] == "self-employed":
            risk_factor += 0.1

        # Age-based adjustments
        if ptype in ["life", "health"]:
            if customer["age"] > 60:
                risk_factor += 0.25
            elif customer["age"] < 25:
                risk_factor += 0.1

        # Premium scaling
        adjusted_premium = int(premium * risk_factor)
        adjusted_premium = max(300, min(adjusted_premium, 5000))  # clamp values

        # Random policy metrics
        satisfaction = round(np.clip(np.random.normal(4.0, 0.6), 2.0, 5.0), 2)
        policy_risk = random.choice(["low", "medium", "high"])
        policy_status = "active" if active else "inactive"

        policies.append({
            "policy_id": policy_id,
            "customer_id": customer["customer_id"],
            "policy_type": ptype,
            "provider": provider,
            "premium_amount": adjusted_premium,
            "coverage_amount": coverage_amount,
            "policy_duration_years": duration_years,
            "policy_start_year": start_year,
            "policy_end_year": end_year,
            "policy_risk_level": policy_risk,
            "policy_status": policy_status,
            "customer_income": customer["annual_income"],
            "customer_credit_score": customer["credit_score"],
            "customer_employment": customer["employment_status"]
        })

    return policies
"""

def generate_policy(customer):
    policies = []
    all_policy_types = ["health", "life", "auto", "home", "travel"]

    coverage_preferences = customer.get("coverage_preferences", all_policy_types)
    policy_types = [p for p in all_policy_types if p in coverage_preferences]

    for ptype in policy_types:
        # Skip incompatible policies
        if ptype == "auto" and not customer.get("vehicle_owner", False):
            continue
        if ptype == "home" and customer.get("existing_loans", "none") == "none":
            continue

        # Base coverage amounts
        base_coverage = {
            "health": 50000,
            "life": 100000,
            "auto": 15000,
            "home": 100000,
            "travel": 50000
        }
        coverage = base_coverage[ptype]

        risk_factor = 1.0

        if ptype in ["health", "life"]:
            if customer["age"] > 60:
                risk_factor += 0.25
                coverage *= 1.2
            elif customer["age"] < 25:
                risk_factor += 0.1
                coverage *= 0.9
            if customer["dependents"] > 0:
                risk_factor += 0.05 * customer["dependents"]
                coverage *= 1 + 0.2 * customer["dependents"]

        if customer["annual_income"] < 40000:
            risk_factor += 0.1
            coverage *= 0.9
        elif customer["annual_income"] > 100000:
            risk_factor -= 0.05
            coverage *= 1.1

        if customer["credit_score"] < 600:
            risk_factor += 0.2
        elif customer["credit_score"] > 750:
            risk_factor -= 0.1

        if ptype in ["life", "health"] and customer["employment_status"] == "self-employed":
            risk_factor += 0.1

        if customer.get("risk_tolerance") == "Low":
            risk_factor *= 0.95
            coverage *= 1.1
        elif customer.get("risk_tolerance") == "High":
            risk_factor *= 1.1
            coverage *= 0.9

        if ptype == "auto":
            coverage *= 1.5 if customer.get("vehicle_type") == "car" else 0.7
        if ptype == "home":
            coverage *= 1.5

        tags = customer.get("interest_tags", [])
        if "travel" in tags and ptype == "travel":
            risk_factor *= 1.05
            coverage *= 1.1
        if "vehicle" in tags and ptype == "auto":
            risk_factor *= 0.95
        if "family" in tags and ptype in ["health", "life"]:
            risk_factor *= 1.05
            coverage *= 1.1

        base_premium = random.randint(500, 2500)
        premium = int(base_premium * risk_factor)
        premium = max(300, min(premium, 5000))

        coverage = int(np.clip(coverage, 5000, 1000000))

        active = random.random() < 0.85
        start_year = random.randint(2015, 2024)
        end_year = start_year + random.randint(1, 10) if not active else start_year + random.randint(1, 10)
        policy_status = "active" if active else "inactive"
        policy_risk = random.choice(["low", "medium", "high"])

        reason_parts = []
        if ptype in ["health", "life"] and customer["dependents"] > 0:
            reason_parts.append("has dependents")
        if ptype == "auto" and customer.get("vehicle_owner", False):
            reason_parts.append("owns a vehicle")
        if ptype in ["health", "life"] and "family" in tags:
            reason_parts.append("interest in family coverage")
        if ptype == "travel" and "travel" in tags:
            reason_parts.append("interest in travel coverage")
        reason = "; ".join(reason_parts) if reason_parts else "Matches customer preferences and budget"

        policies.append({
            "policy_id": f"POL_{random.randint(100000, 999999)}",
            "customer_id": customer["customer_id"],
            "policy_type": ptype,
            "provider": fake.company(),
            "premium_amount": premium,
            "coverage_amount": coverage,
            "policy_duration_years": end_year - start_year,
            "policy_start_year": start_year,
            "policy_end_year": end_year,
            "policy_risk_level": policy_risk,
            "policy_status": policy_status,
            "customer_income": customer["annual_income"],
            "customer_credit_score": customer["credit_score"],
            "customer_employment": customer["employment_status"],
            # "rationale": reason
        })

    return policies

    
    
