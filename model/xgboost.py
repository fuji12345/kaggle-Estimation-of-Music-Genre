import xgboost as xgb


class XGBoost:
    def __init__(self) -> None:
        self.xgb = xgb.XGBClassifier()

    def fit(self, X, y):
        self.xgb.fit(X, y)

    def predict(self, X):
        predict = self.xgb.predict(X)
        return predict
