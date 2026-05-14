# app/explanations.py
from __future__ import annotations

from typing import Any


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

    if feature in ["person_age", "person_emp_exp"]:
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


def build_reason(feature: str, value: Any, recommendation: str) -> str:
    formatted_value = format_feature_value(feature, value)

    if feature == "person_income":
        if recommendation == "Approve":
            return "The applicant’s income level supported the recommendation."
        return "The applicant’s income level increased financial risk."

    if feature == "loan_int_rate":
        if recommendation == "Approve":
            return "The interest rate supported the recommendation."
        return "The interest rate increased the estimated repayment burden."

    if feature == "credit_score":
        if recommendation == "Approve":
            return "The applicant’s credit score supported approval."
        return "The applicant’s credit score increased repayment risk."

    if feature == "previous_loan_defaults_on_file":
        if str(value).strip().lower() == "yes":
            return "Previous repayment problems increased the estimated loan risk."
        return "No previous repayment problems supported approval."

    if feature == "loan_amnt":
        if recommendation == "Approve":
            return "The requested loan amount supported the recommendation."
        return "The requested loan amount increased repayment risk."

    if feature == "person_emp_exp":
        if recommendation == "Approve":
            return "Employment experience supported the recommendation."
        return "Limited employment experience increased uncertainty."

    if feature == "person_home_ownership":
        return f"Housing situation ({formatted_value}) influenced the recommendation."

    if feature == "loan_intent":
        return f"The loan purpose ({formatted_value}) influenced the recommendation."

    if feature == "person_education":
        return f"Education level ({formatted_value}) influenced the recommendation."

    if feature == "person_age":
        return "Applicant age influenced the recommendation."

    return f"{feature.replace('_', ' ').title()} influenced the recommendation."