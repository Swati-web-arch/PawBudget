from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# CORS Headers Middleware
# -----------------------------
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# -----------------------------
# Load Reference Data
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

pet_data_path = os.path.join(base_dir, 'PetData.csv')
if os.path.exists(pet_data_path):
    pet_data = pd.read_csv(pet_data_path)
    BREEDS = sorted(
        pet_data['Breed']
        .dropna()
        .astype(str)
        .unique()
    )
else:
    BREEDS = ["Labrador Retriever", "German Shepherd", "Golden Retriever", "Bulldog", "Beagle", "Poodle", "Unknown"]

# -----------------------------
# Load Original PawBudget Models
# -----------------------------
with open(os.path.join(base_dir, 'cost_model.pkl'), 'rb') as f:
    cost_model = pickle.load(f)

with open(os.path.join(base_dir, 'risk_model.pkl'), 'rb') as f:
    risk_model = pickle.load(f)

with open(os.path.join(base_dir, 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

# -----------------------------
# Load Claims / Medical Cost Models
# -----------------------------
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="swatiyadav123/PawBudget",
    filename="claims_probability_model.pkl",
    repo_type="model"
)

claims_probability_model = joblib.load(model_path)

claims_probability_model = joblib.load(model_path)

medical_cost_model = joblib.load(
    os.path.join(base_dir, 'medical_cost_model.pkl')
)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'PawBudget Integration Engine',
        'models_loaded': True
    })


@app.route('/breeds')
def breeds():
    return jsonify(BREEDS)


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.get_json(silent=True) or {}

    # -----------------------------
    # Basic pet information
    # -----------------------------
    species = data.get('species', 'Dog')
    breed_size = data.get('breed_size', 'Large')
    try:
        age = int(data.get('age', 2))
    except (ValueError, TypeError):
        age = 2
    location = data.get('location', 'Urban')

    # Safe validation against encoders
    if species not in encoders['species'].classes_:
        species = 'Dog'
    if breed_size not in encoders['breed_size'].classes_:
        breed_size = 'Large'
    if location not in encoders['location'].classes_:
        location = 'Urban'

    # -----------------------------
    # Insurance information
    # -----------------------------
    breed = data.get('breed', 'Unknown')
    try:
        premium = float(data.get('premium', 0))
    except (ValueError, TypeError):
        premium = 0.0

    try:
        deductible = float(data.get('deductible', 0))
    except (ValueError, TypeError):
        deductible = 0.0

    enroll_path = data.get('enroll_path', 'Online')

    # Convert website values to the values used during model training
    enroll_path_mapping = {
        'Online': 'Web',
        'Veterinarian': 'Phone',
        'Employer': 'EB',
        'Web': 'Web',
        'Phone': 'Phone',
        'EB': 'EB',
        'Unknown': 'Web'
    }

    enroll_path = enroll_path_mapping.get(enroll_path, 'Web')

    # -----------------------------
    # Previous claims (defaults to 0 for new pet)
    # -----------------------------
    try:
        prev_claim_count = int(data.get('prev_claim_count', 0))
    except (ValueError, TypeError):
        prev_claim_count = 0

    try:
        prev_total_claim_amount = float(data.get('prev_total_claim_amount', 0))
    except (ValueError, TypeError):
        prev_total_claim_amount = 0.0

    try:
        prev_avg_claim_amount = float(data.get('prev_avg_claim_amount', 0))
    except (ValueError, TypeError):
        prev_avg_claim_amount = 0.0

    try:
        prev_max_claim_amount = float(data.get('prev_max_claim_amount', 0))
    except (ValueError, TypeError):
        prev_max_claim_amount = 0.0

    try:
        prev_claimed_days = int(data.get('prev_claimed_days', 0))
    except (ValueError, TypeError):
        prev_claimed_days = 0

    # =====================================================
    # 1. ORIGINAL PAWBUDGET CORE MODEL
    # =====================================================
    encoded_dict = {
        'species': [encoders['species'].transform([species])[0]],
        'breed_size': [encoders['breed_size'].transform([breed_size])[0]],
        'age': [age],
        'location': [encoders['location'].transform([location])[0]]
    }
    encoded_df = pd.DataFrame(encoded_dict)

    pred_cost = float(cost_model.predict(encoded_df)[0])
    pred_risk = str(risk_model.predict(encoded_df)[0])

    # Dynamic Cost breakdown
    food_cost = round(pred_cost * 0.55)
    preventive_cost = round(pred_cost * 0.25)
    grooming_cost = round(pred_cost * 0.20)

    # Scenario simulator
    scenarios = {
        'routine': round(pred_cost),
        'chronic': round(pred_cost * 1.35),
        'emergency': round(pred_cost * 1.75)
    }

    # =====================================================
    # 2. CLAIMS / MEDICAL COST MODEL PIPELINE
    # =====================================================
    claim_input = pd.DataFrame([{
        'Species': species,
        'Breed': breed,
        'PetAgeYears': age,
        'Premium': premium,
        'Deductible': deductible,
        'EnrollPath': enroll_path,
        'PrevClaimCount': prev_claim_count,
        'PrevTotalClaimAmount': prev_total_claim_amount,
        'PrevAvgClaimAmount': prev_avg_claim_amount,
        'PrevMaxClaimAmount': prev_max_claim_amount,
        'PrevClaimedDays': prev_claimed_days
    }])

    # Probability of making a claim
    claim_probability = float(
        claims_probability_model.predict_proba(claim_input)[0][1]
    )

    # Expected medical claim cost
    expected_medical_cost = float(
        medical_cost_model.predict(claim_input)[0]
    )
    expected_medical_cost = max(0.0, expected_medical_cost)

    # =====================================================
    # 3. COMBINED PAWBUDGET FINANCIAL PREDICTION
    # =====================================================
    combined_cost = pred_cost + expected_medical_cost

    financial_risk_score = (
        (claim_probability * 100.0) * 0.6
        + (pred_cost / 10000.0) * 0.4
    )
    financial_risk_score = min(100.0, max(0.0, financial_risk_score))

    if financial_risk_score >= 60:
        financial_risk = 'HIGH'
    elif financial_risk_score >= 35:
        financial_risk = 'MODERATE'
    else:
        financial_risk = 'LOW'

    # =====================================================
    # RETURN RESULTS
    # =====================================================
    return jsonify({
        'ownership_cost': round(pred_cost),
        'claim_probability': round(claim_probability * 100.0, 1),
        'expected_medical_cost': round(expected_medical_cost),
        'combined_cost': round(combined_cost),
        'financial_risk': financial_risk,
        'financial_risk_score': round(financial_risk_score, 1),
        'risk_tier': pred_risk,
        'breakdown': {
            'food': food_cost,
            'preventive': preventive_cost,
            'grooming': grooming_cost
        },
        'scenarios': scenarios
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🐾 PawBudget engine running on http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
