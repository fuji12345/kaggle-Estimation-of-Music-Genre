import xgboost as xgb


class XGBoost:
    def __init__(self) -> None:
        self.model = xgb.XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def predict_proba(self, X):
        predict_proba = self.model.predict_proba(X)
        return predict_proba

    def set_params(self, params: dict):
        self.model.set_params(**params)
