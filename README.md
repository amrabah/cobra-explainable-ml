# COBRA — Explainable ML for Industrial Obsolescence Risk

Portfolio excerpt from an applied AI research project on electronic-component obsolescence management.

The project explores a pipeline combining **domain-informed weak labels, XGBoost and SHAP** to support interpretable obsolescence-risk assessment under incomplete industrial data.

## What this repository highlights

- ML problem formulation for industrial obsolescence risk
- Weak-supervision / weak-label strategy
- Comparison with simple ML baselines
- XGBoost classification
- Explainability-oriented model design
- Backend/frontend architecture used around the ML work
- Explicit treatment of methodological limitations

## ML pipeline

```text
Industrial component data
        |
        v
Domain-informed risk rules
        |
        v
     Weak labels
        |
        v
Baseline models + XGBoost
        |
        v
Explainability / SHAP analysis
```

The original prototype also included PCN/PDN processing, component and nomenclature management, a FastAPI backend, and a React interface. This public portfolio repository intentionally contains a **selected, cleaned excerpt** rather than the complete internal application.

## Methodological note

The labels used for training are weak labels derived from domain rules that use closely related input variables. Therefore, high predictive accuracy against these labels should **not** be interpreted as independent validation of the labeling strategy.

In this setting, performance primarily shows how well the model learns the weak-label mapping. Independent expert labels or observed real-world outcomes would be needed to validate predictive performance externally.

Keeping this limitation explicit was an important part of the research approach.

## Repository structure

```text
backend/
  app/
    modele_ml.py        # selected ML pipeline excerpt
  requirements.txt

frontend/
  index.html
  package.json
  src/
    main.jsx            # frontend entry point excerpt
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · FastAPI · SQLAlchemy · Pydantic · React · Vite

## Research context

Related publications:

- A. Mrabah, E. Saad, M. Zolghadri, C. Edouard (2026), **Explainable AI for Obsolescence Using Weak Labels**, RAMS.
- E. Saad, A. Mrabah, M. Besbes, M. Zolghadri, et al. (2025), **Zero-Shot Learning for Obsolescence Risk Forecasting**, IFAC-PapersOnLine.
- M. Besbes, P. Leclaire, A. Souifi, A. Mrabah, M. Zolghadri (2024), **A New Tool for Obsolescence Management**, CPI 2024.

## Author

**Aya Mrabah** — AI Engineer / Data Scientist

Interests: applied machine learning, explainable AI, NLP and robust ML systems.

---

This repository is shared as a portfolio/research excerpt. It is not a commercial release of COBRA and is not presented as a validated production risk model.
