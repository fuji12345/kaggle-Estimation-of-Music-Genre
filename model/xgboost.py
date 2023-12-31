import xgboost as xgb

from .utils import f1_micro


class XGBoost:
    def __init__(self, config, early_stopping_rounds=100) -> None:
        self.model = xgb.XGBClassifier(
            n_estimators=10000,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=f1_micro,
            objective="multi:softmax",
            num_class=8,
            random_state=config.seed,
        )

    def fit(self, X, y, eval_set, verbose=False):
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def predict_proba(self, X):
        predict_proba = self.model.predict_proba(X)
        return predict_proba

    def set_params(self, params: dict):
        self.model.set_params(**params)
