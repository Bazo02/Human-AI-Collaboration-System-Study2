# Human–AI Collaborative Loan Decision Support System — Study 2

## Overview

This is the Study 2 replication of the [Study 1 system](https://github.com/Bazo02/Human-AI-Collaboration-System), with three methodological improvements addressing limitations identified in Study 1:

1. **Counterbalanced condition order** — participants are alternately assigned baseline-first (AB) or AI-first (BA) to control for learning effects
2. **Embedded explanations** — SHAP-based feature explanations are always visible alongside the AI recommendation, replacing the optional toggle from Study 1
3. **Improved AI model** — XGBoost with SHAP (TreeExplainer) replaces logistic regression, providing more calibrated confidence scores and per-case explanations

The dataset is also changed from the loan approval dataset used in Study 1 to the **Loan Approval Classification dataset**, which produces more realistic uncertainty in the AI model and better borderline cases for participants to evaluate.

Participants complete 24 credit decisions:

* **12 decisions without AI support**
* **12 decisions with AI support** (recommendation + confidence + embedded SHAP explanation)

The order of the two blocks varies between participants based on counterbalancing.

The system measures:

* Decision accuracy (paired t-test, Cohen's dz)
* Decision time
* Trust and perceived usability (SUS)
* AI reliance behavior (AI-followed rate, distribution)
* Spearman correlation between trust and AI-followed rate
* Order effects (AB vs BA subgroup comparison)

---

## Differences from Study 1

| | Study 1 | Study 2 |
|---|---|---|
| Condition order | Baseline always first | Counterbalanced (AB / BA) |
| Explanations | Optional toggle (hidden by default) | Always visible (embedded) |
| AI model | Logistic regression | XGBoost |
| Explanations method | Linear coefficients (x_i × coef_i) | SHAP TreeExplainer (per-case) |
| Dataset | Loan approval (Kaggle) | Loan Approval Classification |
| AI confidence | High and stable (~0.875) | Lower and more varied confidence scores |
| Borderline cases | Limited | Expanded borderline selection |
| Model test accuracy | ~0.78 | 0.928 |

---

## Tech Stack

* Python (Flask)
* scikit-learn
* XGBoost
* SHAP (TreeExplainer)
* pandas
* matplotlib (server-side backend)
* SQLite
* HTML / CSS / JavaScript

---

## Installation

```bash
python -m venv venv

source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## Run Locally

Run the following commands from the project root directory:

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

python -m app.data_prep
python -m app.model_train
python -m app.main
```

Then open:

```
http://127.0.0.1:5000
```

**Important:** Run commands from the project root folder, not inside `/app`.

---

## Data

All study data is stored in:

```
/outputs/study.db
```

### Tables

* `participants` — one row per participant, includes `condition_order` (AB or BA)
* `decisions` — one row per case decision
* `events` — interaction events with JSON payload
* `surveys` — post-task questionnaire answers

---

## Research Purpose

Study 2 extends Study 1 by addressing three methodological limitations. It enables controlled evaluation of:

* Whether counterbalancing affects accuracy and decision time outcomes
* Whether embedded (always-visible) explanations increase engagement with AI reasoning compared to optional explanations
* Whether a more accurate and better-calibrated AI model changes reliance patterns
* Trust and reliance in Human–AI decision support systems