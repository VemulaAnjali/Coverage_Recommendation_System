# app.py
from flask import Flask, render_template, request
from llm_extractor import extract_customer_from_conversation
from recommendation import normalize_customer_input, recommend_policies_with_agents
from mapping import build_new_customer
from recommender import recommend_policies

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    extracted = None
    customer_profile = None
    error = None

    if request.method == "POST":
        transcript = request.form.get("transcript", "").strip()
        top_n = request.form.get("top_n", "2")

        try:
            top_n = int(top_n)
        except ValueError:
            top_n = 2

        if not transcript:
            error = "Please paste a transcript."
        else:
            try:
                extracted = extract_customer_from_conversation(transcript)
                # customer_profile = build_new_customer(extracted)
                # recommendations = recommend_policies(customer_profile, top_n=top_n)
                new_customer = normalize_customer_input(extracted)
                recommendations = recommend_policies_with_agents(new_customer, top_n=top_n)

            except Exception as e:
                error = f"Something went wrong: {e}"

    return render_template(
        "index.html",
        recommendations=recommendations,
        extracted=extracted,
        # customer_profile=customer_profile,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
