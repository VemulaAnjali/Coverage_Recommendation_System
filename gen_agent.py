import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker("en_US")
random.seed(42)
np.random.seed(42)

def generate_agents(n=50):
    regions = ["California", "Texas", "Florida", "New York", "Illinois", "Washington", "Minnesota"]
    specializations = ["auto", "life", "health", "home", "travel"]
    
    agents = []
    
    for i in range(n):
        agent_id = f"AGT_{1000 + i}"
        name = fake.name()
        region = random.choice(regions)
        specialization = random.choice(specializations)
        experience = random.randint(1, 20)
        success_rate = round(np.clip(np.random.normal(0.75, 0.1), 0.4, 0.95), 2)
        avg_rating = round(np.clip(np.random.normal(4.2, 0.5), 2.5, 5.0), 2)
        active_clients = random.randint(20, 300)
        languages = random.sample(["English", "Spanish", "French", "German", "Mandarin"], k=random.randint(1, 3))
        contact = fake.phone_number()
        email = f"{name.split()[0].lower()}.{name.split()[-1].lower()}@insureplus.com"

        agents.append({
            "agent_id": agent_id,
            "agent_name": name,
            "region": region,
            "specialization": specialization,
            "experience_years": experience,
            "success_rate": success_rate,
            "average_rating": avg_rating,
            "active_clients": active_clients,
            "languages": ", ".join(languages),
            "contact_number": contact,
            "email": email
        })
    
    return pd.DataFrame(agents)
