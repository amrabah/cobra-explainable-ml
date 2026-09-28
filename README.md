# COBRA — Explainable ML for Industrial Obsolescence Risk

An end-to-end prototype for industrial electronic-component obsolescence management, combining structured data ingestion, PCN/PDN processing, risk assessment, weak supervision, machine learning and explainability.

## Why this project

Electronic-component obsolescence creates supply-chain and engineering risk: parts may become difficult to source, be discontinued, or require redesign. COBRA explores how data-driven methods can support engineers by consolidating heterogeneous information and producing interpretable risk assessments.

This repository is a portfolio-oriented prototype extracted from work conducted in the context of the COBRA / EOS research project. The included example data are synthetic and intended for demonstration.

## What it demonstrates

- End-to-end ML workflow, from data preparation to model interpretation
- Weak-label generation for an industrial risk problem
- XGBoost-based classification and SHAP explainability
- FastAPI backend with SQLAlchemy and Pydantic
- React/Vite frontend
- PCN/PDN ingestion and parsing
- Component criticality and obsolescence-risk services
- Automated tests and synthetic demonstration data

## Architecture

```text
Synthetic / imported component data
            |
            v
     Data ingestion layer
            |
            +--> PCN / PDN processing
            |
            +--> Criticality & risk rules
            |
            v
       Weak labels
            |
            v
        XGBoost model
            |
            v
      SHAP explanations
            |
            v
 FastAPI backend <--> React frontend
```

## Repository structure

```text
backend/
  app/
    main.py                 # FastAPI application
    database.py             # Database configuration
    models.py / schemas.py  # Data model and validation
    smartpcn.py             # PCN/PDN processing
    criticite.py            # Criticality logic
    risque.py               # Risk logic
    modele_ml.py            # ML pipeline and explainability
    services*.py            # Application services
  samples/                  # Synthetic/example PCN and XML files
  test_*.py                 # Tests

frontend/
  src/
    App.jsx
    api.js
    apiRisque.js
    styles.css

DOCUMENTATION.md
DOCUMENTATION-RISQUE.md
```

## Machine-learning approach

The prototype uses weak supervision to derive training labels from domain-informed risk signals, then trains an XGBoost classifier to learn the resulting risk structure. SHAP is used to expose feature contributions and make individual predictions easier to inspect.

### An important methodological limitation

High predictive performance against weak labels does **not** independently validate the labeling strategy when those labels were themselves derived from the same or closely related input variables. In that setting, the model may primarily reproduce the labeling function.

For this reason, model metrics in this prototype should be interpreted as evidence that the model can learn the weak-label mapping — not as proof of real-world predictive validity. A production evaluation would require independent outcomes or expert-validated labels.

This distinction was important in the research work behind the prototype and is intentionally kept visible here.

## Tech stack

**Machine learning:** Python, XGBoost, SHAP, scikit-learn, pandas, NumPy  
**Backend:** FastAPI, SQLAlchemy, Pydantic  
**Frontend:** React, Vite  
**Engineering:** Git, API-based architecture, automated tests

## Running locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Exact configuration may need to be adapted to your local database/environment.

## Research context

The prototype was developed in the context of research on AI-assisted obsolescence management. Related work includes:

- A. Mrabah, E. Saad, M. Zolghadri, C. Edouard, **“Explainable AI for Obsolescence Using Weak Labels,”** RAMS 2026.
- E. Saad, A. Mrabah, M. Besbes, M. Zolghadri, et al., **“Zero-Shot Learning for Obsolescence Risk Forecasting,”** IFAC-PapersOnLine, 2025.
- M. Besbes, P. Leclaire, A. Souifi, A. Mrabah, M. Zolghadri, **“A New Tool for Obsolescence Management,”** CPI 2024.

## Author

**Aya Mrabah** — AI Engineer / Data Scientist

Research interests include applied machine learning, explainable AI, NLP and robust ML systems.

## Notes

This repository is presented as a research and engineering prototype. It is not a commercial release of COBRA and should not be interpreted as a validated production risk model.
