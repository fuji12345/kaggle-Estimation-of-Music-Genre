import xgboost as xgb


class XGBoost:
    def __init__(self) -> None:
        self.model = xgb.XGBClassifier(n_estimators=10000, early_stopping_rounds=30, eval_metric="auc")

    def fit(self, X, y, eval_set):
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def set_params(self, params: dict):
        self.model.set_params(**params)
