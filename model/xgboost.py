import xgboost as xgb


class XGBoost:
    def __init__(self) -> None:
        self.model = xgb.XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def set_params(self, params: dict):
        self.model.set_params(**params)
