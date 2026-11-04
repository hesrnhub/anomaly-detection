"""Daftar 6 algoritma representatif, hiperparameter default."""
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.xgbod import XGBOD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# nama -> (tingkat supervisi, pembuat model)
MODELS = {
    "IForest": ("unsupervised", lambda seed: IForest(random_state=seed)),
    "OCSVM": ("unsupervised", lambda seed: OCSVM()),
    "ECOD": ("unsupervised", lambda seed: ECOD()),
    "XGBOD": ("semi-supervised", lambda seed: XGBOD(random_state=seed)),
    "RandomForest": ("supervised", lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1)),
    "XGBoost": ("supervised", lambda seed: XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1)),
}


def score(model, name, X):
    """Skor anomali: makin besar makin anomali."""
    if name in ("RandomForest", "XGBoost"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
