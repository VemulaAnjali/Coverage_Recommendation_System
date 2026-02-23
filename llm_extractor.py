# llm_extractor.py
import os
import json
from mistralai import Mistral
from dotenv import load_dotenv


load_dotenv(override=True) 

MISTRAL_MODEL = "mistral-large-2411"

def use_mistral(prompt, model=MISTRAL_MODEL, op_type="json_object"):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set. Check your .env file.")

    client = Mistral(api_key=api_key)

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        response_format={"type": op_type},
    )

    return response.choices[0].message.content


def get_prompt(conversation: str) -> str:
    keyword_extraction_prompt = f"""
You are an expert Insurance Agent. You will be given a conversation between an agent and a customer.
Your task is to extract all the customer details from the conversation.

If a field is not mentioned in the conversation, set its value to "nan" (string).

Here is the conversation:
{conversation}

Return JSON exactly in this schema:

{{
  "name": "",
  "phone": "",
  "city": "",
  "state": "",
  "zipcode": "",
  "age": "",
  "gender": "",
  "marital_status": "",
  "dependents": "",
  "education_level": "",
  "occupation": "",
  "employment_status": "",
  "annual_income": "",
  "credit_score": "",
  "existing_loans": "",
  "avg_monthly_expense": "",
  "vehicle_owner": "",
  "vehicle_type": "",
  "budget_per_month": "",
  "customer_tenure_years": "",
  "risk_tolerance": ""
}}

Keep your output strictly in valid JSON — no other text.
"""
    return keyword_extraction_prompt


def extract_customer_from_conversation(conversation: str) -> dict:
    prompt = get_prompt(conversation)
    json_str = use_mistral(prompt)
    data = json.loads(json_str)
    return data
