import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print("Loading datasets...")

pets = pd.read_csv("PetData.csv")
claims = pd.read_csv("ClaimData.csv")

# Convert dates
pets["EnrollDate"] = pd.to_datetime(pets["EnrollDate"])
claims["ClaimDate"] = pd.to_datetime(claims["ClaimDate"])

# Join claims with pet information
data = claims.merge(
    pets[["PetId", "Species", "Breed", "PetAge", "Premium",
          "Deductible", "EnrollPath", "EnrollDate"]],
    on="PetId",
    how="inner"
)

# Days since enrollment
data["DaysSinceEnrollment"] = (
    data["ClaimDate"] - data["EnrollDate"]
).dt.days

# Keep claims during the first two policy years
data = data[
    (data["DaysSinceEnrollment"] >= 0) &
    (data["DaysSinceEnrollment"] < 730)
].copy()

# Create first-year and second-year totals
first_year = data[data["DaysSinceEnrollment"] < 365]
second_year = data[
    (data["DaysSinceEnrollment"] >= 365) &
    (data["DaysSinceEnrollment"] < 730)
]

# First-year features
first_features = first_year.groupby("PetId").agg(
    PrevClaimCount=("AmountClaimed", "count"),
    PrevTotalClaimAmount=("AmountClaimed", "sum"),
    PrevAvgClaimAmount=("AmountClaimed", "mean"),
    PrevMaxClaimAmount=("AmountClaimed", "max")
).reset_index()

# Number of different days with claims
claimed_days = (
    first_year.groupby("PetId")["ClaimDate"]
    .nunique()
    .reset_index(name="PrevClaimedDays")
)

first_features = first_features.merge(
    claimed_days,
    on="PetId",
    how="left"
)

# Second-year target
second_target = (
    second_year.groupby("PetId")["AmountClaimed"]
    .sum()
    .reset_index(name="NextTotalClaimAmount")
)

# Start with all pets
model_data = pets[
    ["PetId", "Species", "Breed", "PetAge", "Premium",
     "Deductible", "EnrollPath"]
].copy()

model_data = model_data.merge(
    first_features,
    on="PetId",
    how="left"
)

model_data = model_data.merge(
    second_target,
    on="PetId",
    how="left"
)

# Pets with no first-year claims get zero
claim_features = [
    "PrevClaimCount",
    "PrevTotalClaimAmount",
    "PrevAvgClaimAmount",
    "PrevMaxClaimAmount",
    "PrevClaimedDays"
]

model_data[claim_features] = model_data[claim_features].fillna(0)

# Pets with no second-year claims have zero cost
model_data["NextTotalClaimAmount"] = (
    model_data["NextTotalClaimAmount"].fillna(0)
)

# Convert PetAge into approximate years
age_map = {
    "8 weeks to 12 months old": 0.5,
    "1 year old": 1,
    "2 years old": 2,
    "3 years old": 3,
    "4 years old": 4,
    "5 years old": 5,
    "6 years old": 6,
    "7 years old": 7,
    "8 years old": 8,
    "9 years old": 9,
    "10 years old": 10,
    "11 years old": 11,
    "12 years old": 12,
    "13 years old": 13,
    "14 years old": 14,
    "15 years old": 15,
    "16 years old": 16
}

model_data["PetAgeYears"] = (
    model_data["PetAge"]
    .map(age_map)
    .fillna(5)
)

# Features
features = [
    "Species",
    "Breed",
    "PetAgeYears",
    "Premium",
    "Deductible",
    "EnrollPath",
    "PrevClaimCount",
    "PrevTotalClaimAmount",
    "PrevAvgClaimAmount",
    "PrevMaxClaimAmount",
    "PrevClaimedDays"
]

X = model_data[features]

# Target 1: whether the pet makes a claim next year
y_claim = (
    model_data["NextTotalClaimAmount"] > 0
).astype(int)

# Target 2: total medical cost next year
y_cost = model_data["NextTotalClaimAmount"]

# Categorical and numerical columns
categorical = [
    "Species",
    "Breed",
    "EnrollPath"
]

numerical = [
    "PetAgeYears",
    "Premium",
    "Deductible",
    "PrevClaimCount",
    "PrevTotalClaimAmount",
    "PrevAvgClaimAmount",
    "PrevMaxClaimAmount",
    "PrevClaimedDays"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        ),
        (
            "num",
            "passthrough",
            numerical
        )
    ]
)

# -------------------------------
# CLAIM PROBABILITY MODEL
# -------------------------------

print("Training claim probability model...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_claim,
    test_size=0.2,
    random_state=42,
    stratify=y_claim
)

claim_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )
])

claim_model.fit(X_train, y_train)

joblib.dump(
    claim_model,
    "claims_probability_model.pkl"
)

print("Saved claims_probability_model.pkl")


# -------------------------------
# MEDICAL COST MODEL
# -------------------------------

print("Training medical cost model...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cost,
    test_size=0.2,
    random_state=42
)

cost_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )
    )
])

cost_model.fit(X_train, y_train)

joblib.dump(
    cost_model,
    "medical_cost_model.pkl"
)

print("Saved medical_cost_model.pkl")

print()
print("===================================")
print("TRAINING COMPLETE!")
print("===================================")
print("claims_probability_model.pkl")
print("medical_cost_model.pkl")