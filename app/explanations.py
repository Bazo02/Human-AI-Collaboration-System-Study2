# app/explanations.py

from __future__ import annotations

from typing import Any


# Formats a raw feature value into a readable string for display
def format_feature_value(feature: str, value: Any) -> str:
    if value is None:
        return ""

    if feature in ["person_income", "loan_amnt"]:
        try:
            return f"${float(value):,.0f}"
        except Exception:
            return str(value)

    if feature == "loan_int_rate":
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return str(value)

    if feature in ["person_age", "person_emp_exp", "credit_score"]:
        try:
            return f"{int(float(value))}"
        except Exception:
            return str(value)

    text = str(value).strip()

    if text.upper() == "YES":
        return "Yes"

    if text.upper() == "NO":
        return "No"

    if feature == "loan_intent":
        mapping = {
            "HOMEIMPROVEMENT": "Home improvement",
            "VENTURE": "Business investment",
            "EDUCATION": "Education",
            "MEDICAL": "Medical expenses",
            "PERSONAL": "Personal expenses",
            "DEBTCONSOLIDATION": "Debt consolidation",
        }
        return mapping.get(text.upper(), text.title())

    if feature == "person_home_ownership":
        mapping = {
            "RENT": "Renting",
            "OWN": "Owns home",
            "MORTGAGE": "Mortgage",
            "OTHER": "Other",
        }
        return mapping.get(text.upper(), text.title())

    return text.replace("_", " ").title()


# Builds a short human-readable phrase describing a feature and its value
def _feature_phrase(feature: str, formatted_value: str) -> str:
    if feature == "person_income":
        return f"Annual income: {formatted_value}"

    if feature == "loan_int_rate":
        return f"Interest rate: {formatted_value}"

    if feature == "credit_score":
        return f"Credit score: {formatted_value}"

    if feature == "previous_loan_defaults_on_file":
        return f"Previous repayment problems: {formatted_value}"

    if feature == "loan_amnt":
        return f"Requested loan amount: {formatted_value}"

    if feature == "person_emp_exp":
        return f"Employment experience: {formatted_value} years"

    if feature == "person_home_ownership":
        return f"Housing situation: {formatted_value}"

    if feature == "loan_intent":
        return f"Loan purpose: {formatted_value}"

    if feature == "person_education":
        return f"Education level: {formatted_value}"

    if feature == "person_age":
        return f"Applicant age: {formatted_value} years"

    if formatted_value:
        return f"{feature.replace('_', ' ').title()}: {formatted_value}"

    return feature.replace("_", " ").title()


# Builds the full explanation sentence shown to the user for one feature
def build_reason(
    feature: str,
    value: Any,
    recommendation: str,
    contribution: float,
) -> str:
    formatted_value = format_feature_value(feature, value)
    phrase = _feature_phrase(feature, formatted_value)

    return (
        f"{phrase}. This was one of the main factors used by the AI model "
        f"when calculating this recommendation."
    )
