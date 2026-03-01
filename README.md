# Coverage Recommendation System

An AI-powered insurance recommendation system that analyzes agent–customer call transcripts to generate personalized policy recommendations and agent suggestions.

Built as part of **CSCI 5707 - Principles of Database Systems** at the University of Minnesota.

## Overview

Customers often struggle to navigate complex insurance policies. This system extracts structured information from call transcripts using an LLM, clusters customers into behavioral segments, and recommends the most suitable insurance plans using cosine similarity, all from a single conversation.

## Architecture

The pipeline consists of four major components:

1. **Keyword Extraction (LLM)** — Zero-shot prompting with Mistral Large 2411 converts messy call transcripts into structured JSON (age, income, dependents, location, insurance interests, etc.)
2. **Customer Clustering** — UMAP dimensionality reduction + K-Means segments customers into behavioral personas (budget-conscious, family-oriented, high-income, retirement-age)
3. **Policy Recommendation** — Cosine similarity between the customer's feature vector and averaged policy-holder vectors produces a ranked list of recommended insurance products (auto, life, health, home, travel). A cluster-based recommendation using KNN is also provided.
4. **Agent Suggestion** — Rule-based ranking by geography, specialization, success rate, ratings, and experience

## Tech Stack

- **Python** — Flask, Pandas, Scikit-learn, UMAP, NumPy
- **Mistral Large 2411** — LLM for keyword extraction from transcripts
- **Flask** — Web application framework
- **Scikit-learn** — StandardScaler, OneHotEncoder, KMeans, NearestNeighbors, Cosine Similarity
- **UMAP** — Dimensionality reduction for clustering
- **Faker** — Synthetic data generation


## Getting Started

### Prerequisites

- Python 3.9+
- Mistral API key

### Installation

```bash
git clone https://github.com/<your-username>/coverage-recommendation-system.git
cd coverage-recommendation-system
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
FLASK_ENV=development
FLASK_APP=app.py
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Generate Synthetic Data (if needed)

```bash
python gen_data.py
python clustering.py
```

This creates `customers.csv`, `policies.csv`, `agents.csv`, and `customers_clustered.csv`.

### Run Locally

```bash
flask run
```

Or:

```bash
python app.py
```

The app will be available at `http://localhost:5000`.

## How It Works

1. User pastes a call transcript into the web UI and selects the number of recommendations
2. The transcript is sent to Mistral Large 2411 with a zero-shot prompt to extract structured customer attributes as JSON
3. Attributes are cleaned and normalized (StandardScaler for numeric, OneHotEncoder for categorical)
4. **Cosine similarity** ranks the best-matching policy types by comparing the customer vector against averaged policy-holder vectors
5. **Cluster-based recommendation** assigns the customer to the nearest cluster via KNN and returns the most popular policies in that segment
6. For each recommended policy, an agent is suggested based on location match, specialization, success rate, and rating

## Results

- Mistral reliably produces structured JSON even from noisy, informal transcripts
- Four distinct customer segments were identified through UMAP + K-Means
- Policy recommendations align with industry expectations (e.g., high-income homeowners with dependents → Home + Life)
- Agent suggestions adapt based on geographic and performance-based criteria

## Team

| Name | Email |
|------|-------|
| Anjali Vemula | vemul034@umn.edu |
| Barath Ganesh | ganes159@umn.edu |
| Raghuram Gowrav Madduri | maddu046@umn.edu |
| Reshma Rao Chandukudlu Hosamane | chand950@umn.edu |
