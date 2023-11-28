import optuna
from sklearn.model_selection import cross_val_score

import model


def xgboost_config(trial):
    params_dict = {
        "eta": trial.suggest_float("eta", 1e-4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 1.0),
        "n_estimators": trial.suggest_int(f"n_estimators", 10, 500),
    }
    return params_dict


def get_model_config(model_name, trial):
    if model_name == "XGBoost":
        return xgboost_config(trial)
    else:
        return ValueError()


class Optuna:
    def __init__(self, model_name, X, y, cv=10, n_trials=100) -> None:
        self.model_name = model_name
        self.X, self.y = X, y
        self.cv = cv
        self.n_trials = n_trials
        self.model = getattr(model, self.model_name)()

    def objective(self, trial):
        self.model_params_dict = get_model_config(self.model_name, trial)
        self.model.set_params(self.model_params_dict)
        scores = cross_val_score(self.model.model, self.X, self.y, cv=self.cv, scoring="f1_micro")
        val_score_mean = scores.mean()

        return val_score_mean

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        best_params = study.best_trial.params
        best_score = study.best_trial.value
        print(f"best params: {best_params}\nscore: {best_score}")
        return best_params
