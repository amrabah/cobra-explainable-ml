# -*- coding: utf-8 -*-
"""
ML layer for the COBRA risk-management prototype.

Pipeline:
    domain rules -> weak labels -> XGBoost -> SHAP

Important: labels are derived from closely related input variables. High
accuracy therefore measures the model's ability to reproduce the weak-label
mapping; it is not independent validation of the business labeling strategy.
"""
from __future__ import annotations
import json, warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .risque import BASSE, HAUTE, MOYENNE, risque_global

VARIABLES_NUMERIQUES = [
    "y_to_eol","nb_crosses","inventaire_marche","nb_distributeurs",
    "delai_fabricant_semaines","prix_moyen","age_annees",
]
VARIABLES_CATEGORIELLES = [
    "fabricant","ligne_produit","statut_lifecycle","rohs",
    "reach_svhc","meilleur_code_cross",
]
VARIABLES = VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES
LIBELLES_VARIABLES = {
    "y_to_eol":"Années avant fin de vie (Y-to-EOL)",
    "nb_crosses":"Nombre de remplaçants (Crosses)",
    "inventaire_marche":"Inventaire disponible sur le marché",
    "nb_distributeurs":"Nombre de distributeurs",
    "delai_fabricant_semaines":"Délai fabricant (semaines)",
    "prix_moyen":"Prix moyen","age_annees":"Ancienneté depuis introduction (années)",
    "fabricant":"Fabricant","ligne_produit":"Ligne de produit",
    "statut_lifecycle":"Statut de cycle de vie","rohs":"Statut RoHS",
    "reach_svhc":"REACH — présence de SVHC",
    "meilleur_code_cross":"Meilleure compatibilité de remplacement",
}
CLASSES=[BASSE,MOYENNE,HAUTE]

def _meilleur_code_cross(composant)->Optional[str]:
    from .risque import lire_crosses
    croisements=lire_crosses(composant)
    if not croisements:return None
    codes={c.get("code","").upper() for c in croisements}
    for code in ("A","B","C","D","E"):
        if code in codes:return code
    return None

def ligne_variables(c):
    return {
        "y_to_eol":c.y_to_eol,"nb_crosses":c.nb_crosses,
        "inventaire_marche":c.inventaire_marche,
        "nb_distributeurs":c.nb_distributeurs,
        "delai_fabricant_semaines":c.delai_fabricant_semaines,
        "prix_moyen":c.prix_moyen,"age_annees":c.age_annees,
        "fabricant":c.fabricant or None,"ligne_produit":c.ligne_produit or None,
        "statut_lifecycle":c.statut_lifecycle or None,"rohs":c.rohs or None,
        "reach_svhc":c.reach_svhc or None,
        "meilleur_code_cross":_meilleur_code_cross(c),
    }

def construire_jeu(db):
    from .models import Composant
    lignes,labels,refs=[],[],[]
    for c in db.query(Composant).all():
        niveau=risque_global(c)["niveau"]
        if niveau not in CLASSES:continue
        lignes.append(ligne_variables(c));labels.append(niveau);refs.append(c.reference_fabricant)
    X=pd.DataFrame(lignes,columns=VARIABLES)
    for col in VARIABLES_CATEGORIELLES:X[col]=X[col].astype("category")
    for col in VARIABLES_NUMERIQUES:X[col]=pd.to_numeric(X[col],errors="coerce")
    return X,pd.Series(labels,name="risque_global"),refs

@dataclass
class ModeleRisque:
    modele=None
    explainer=None
    categories:dict=field(default_factory=dict)
    metriques:dict=field(default_factory=dict)
    entraine_le:Optional[str]=None
    nb_exemples:int=0
    @property
    def pret(self):return self.modele is not None
_ETAT=ModeleRisque()

def _aligner_categories(X,categories):
    X=X.copy()
    for col,modalites in categories.items():
        X[col]=pd.Categorical(X[col].astype(object),categories=modalites)
    return X

def entrainer(db,graine=42):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder,StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    import shap
    X,y,_=construire_jeu(db)
    if len(X)<100:raise ValueError("Training set too small.")
    categories={c:list(X[c].cat.categories) for c in VARIABLES_CATEGORIELLES}
    y_num=y.map({n:i for i,n in enumerate(CLASSES)}).astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y_num,test_size=.2,random_state=graine,stratify=y_num)
    def encoder(df):
        d=df.copy()
        for c in VARIABLES_CATEGORIELLES:d[c]=d[c].astype(object).where(d[c].notna(),"Inconnu")
        return d
    enc=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
    imp=SimpleImputer(strategy="median")
    cat_tr=enc.fit_transform(encoder(Xtr)[VARIABLES_CATEGORIELLES])
    cat_te=enc.transform(encoder(Xte)[VARIABLES_CATEGORIELLES])
    num_tr=imp.fit_transform(Xtr[VARIABLES_NUMERIQUES]);num_te=imp.transform(Xte[VARIABLES_NUMERIQUES])
    flat_tr=np.hstack([num_tr,cat_tr]);flat_te=np.hstack([num_te,cat_te])
    results=[]
    def evaluate(name,pred,proba=None):
        item={"modele":name,"exactitude":round(float(accuracy_score(yte,pred))*100,2),
              "f1_macro":round(float(f1_score(yte,pred,average="macro")),3),
              "f1_pondere":round(float(f1_score(yte,pred,average="weighted")),3)}
        results.append(item);return item
    base=DummyClassifier(strategy="most_frequent").fit(flat_tr,ytr)
    evaluate("Majority-class baseline",base.predict(flat_te))
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=graine)).fit(flat_tr,ytr)
    evaluate("Logistic Regression",lr.predict(flat_te))
    dt=DecisionTreeClassifier(random_state=graine,max_depth=8).fit(flat_tr,ytr)
    evaluate("Decision Tree",dt.predict(flat_te))
    rf=RandomForestClassifier(n_estimators=200,random_state=graine,n_jobs=-1).fit(flat_tr,ytr)
    evaluate("Random Forest",rf.predict(flat_te))
    xgb=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.1,subsample=.9,
        colsample_bytree=.9,objective="multi:softprob",num_class=len(CLASSES),
        enable_categorical=True,tree_method="hist",random_state=graine,n_jobs=-1).fit(Xtr,ytr)
    retained=evaluate("XGBoost",xgb.predict(Xte),xgb.predict_proba(Xte))
    _ETAT.modele=xgb;_ETAT.categories=categories;_ETAT.explainer=shap.TreeExplainer(xgb)
    _ETAT.entraine_le=datetime.utcnow().isoformat(timespec="seconds");_ETAT.nb_exemples=len(X)
    _ETAT.metriques={"comparaison":results,"retenu":retained,"nb_exemples":len(X),
      "avertissement":"High accuracy reproduces weak labels; it does not independently validate them."}
    return _ETAT.metriques

def etat():
    return {"entraine":_ETAT.pret,"entraine_le":_ETAT.entraine_le,
            "nb_exemples":_ETAT.nb_exemples,"metriques":_ETAT.metriques or None}

def predire(composant):
    if not _ETAT.pret:return None
    X=pd.DataFrame([ligne_variables(composant)],columns=VARIABLES)
    for c in VARIABLES_NUMERIQUES:X[c]=pd.to_numeric(X[c],errors="coerce")
    X=_aligner_categories(X,_ETAT.categories)
    probas=_ETAT.modele.predict_proba(X)[0];idx=int(np.argmax(probas))
    return {"niveau":CLASSES[idx],"confiance":round(float(probas[idx]),3),
            "probabilites":{c:round(float(probas[i]),3) for i,c in enumerate(CLASSES)}}
