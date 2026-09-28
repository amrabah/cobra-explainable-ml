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

## Product overview

COBRA was designed as a broader decision-support platform rather than an isolated ML notebook. The interface prototypes cover several stages of the obsolescence-management workflow:

**Obsolescence dashboard** — consolidated component status, compliance and inventory indicators, risk levels and operational monitoring in one system view.

**RETEX / recommendation module** — retrieval of similar historical cases, similarity scoring and presentation of previous or alternative mitigation actions. This connects NLP/similarity work with an end-user decision workflow.

**PCN / PDN notifications** — centralized tracking of supplier product-change and product-discontinuation notices, with supplier, category, effective date and criticality information.

**Platform experience** — a unified interface designed to connect component data, risk assessment, notifications and recommendations instead of exposing the underlying models directly to users.

[View the COBRA interface design in Figma](https://www.figma.com/design/fyzli7EbAH6VnqyLgq6yWF/COBRA-project--Copy-?node-id=0-1&p=f)

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

The original prototype also included PCN/PDN processing, component and nomenclature management, a FastAPI backend, and a React interface. This portfolio repository intentionally contains a **selected, cleaned excerpt** rather than the complete internal application.

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
