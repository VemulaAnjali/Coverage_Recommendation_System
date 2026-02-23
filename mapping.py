# mapping.py
from recommender import NUMERIC_COLS, CATEGORICAL_COLS

def _to_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).strip().lower()
    if value in ("", "nan", "none", "null"):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _to_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip().lower()
    if value in ("", "nan", "none", "null"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("yes", "y", "true", "1"):
        return True
    if v in ("no", "n", "false", "0"):
        return False
    return default


def build_new_customer(llm_json: dict) -> dict:
    """
    llm_json: output from extract_customer_from_conversation()
    returns a dict compatible with recommend_policies()
    """

    # Basic defaults (you can tweak these)
    defaults = {
        "age": 35,
        "annual_income": 60000,
        "credit_score": 700,
        "avg_monthly_expense": 1500,
        "budget_per_month": 300,
        "customer_tenure_years": 1,
        "gender": "unknown",
        "marital_status": "unknown",
        "education_level": "unknown",
        "occupation": "unknown",
        "employment_status": "employed",
        "vehicle_owner": False,
        "vehicle_type": "none",
        "risk_tolerance": "Medium",
    }

    def get_field(key):
        val = llm_json.get(key, "nan")
        if isinstance(val, str) and val.strip().lower() == "nan":
            return None
        return val

    new_customer = {
        "customer_id": "CUST_FROM_CALL",
        "name": get_field("name") or "Unknown",
        "phone": get_field("phone") or "",
        "city": get_field("city") or "",
        "state": get_field("state") or "",
        "zipcode": get_field("zipcode") or "",
        "dependents": _to_int(get_field("dependents"), 0),
        "existing_loans": get_field("existing_loans") or "none",
    }

    # Numeric fields
    new_customer["age"] = _to_int(get_field("age"), defaults["age"])
    new_customer["annual_income"] = _to_int(
        get_field("annual_income"), defaults["annual_income"]
    )
    new_customer["credit_score"] = _to_int(
        get_field("credit_score"), defaults["credit_score"]
    )
    new_customer["avg_monthly_expense"] = _to_int(
        get_field("avg_monthly_expense"), defaults["avg_monthly_expense"]
    )
    new_customer["budget_per_month"] = _to_int(
        get_field("budget_per_month"), defaults["budget_per_month"]
    )
    new_customer["customer_tenure_years"] = _to_float(
        get_field("customer_tenure_years"), defaults["customer_tenure_years"]
    )

    # Categorical fields expected by recommender
    new_customer["gender"] = (get_field("gender") or defaults["gender"]).lower()
    new_customer["marital_status"] = (get_field("marital_status") or defaults["marital_status"]).lower()
    new_customer["education_level"] = (get_field("education_level") or defaults["education_level"]).lower()
    new_customer["occupation"] = (get_field("occupation") or defaults["occupation"]).lower()
    new_customer["employment_status"] = (get_field("employment_status") or defaults["employment_status"]).lower()

    new_customer["vehicle_owner"] = _to_bool(get_field("vehicle_owner"), defaults["vehicle_owner"])
    new_customer["vehicle_type"] = (get_field("vehicle_type") or defaults["vehicle_type"]).lower()
    new_customer["risk_tolerance"] = (get_field("risk_tolerance") or defaults["risk_tolerance"]).capitalize()

    return new_customer
