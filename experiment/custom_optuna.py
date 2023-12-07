import model
import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def xgboost_config(trial):
    params_dict = {
        "eta": trial.suggest_float("eta", 0.001, 0.1),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 0.1, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 100.0),
        "alpha": trial.suggest_float("alpha", 0.0, 100.0),
    }
    return params_dict


def get_model_config(model_name, trial):
    if model_name == "XGBoost":
        return xgboost_config(trial)
    else:
        return ValueError()


class Optuna:
    def __init__(self, model_name, X, y, cv, n_trials) -> None:
        self.model_name = model_name
        self.X, self.y = X, y

        self.cv = cv
        self.n_trials = n_trials
        self.model = getattr(model, self.model_name)()

    def objective(self, trial):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        scores = np.zeros(self.cv)
        for i_fold, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
            train_X, train_y = self.X.iloc[train_index], self.y.iloc[train_index]
            val_X, val_y = self.X.iloc[val_index], self.y.iloc[val_index]

            self.model_params_dict = get_model_config(self.model_name, trial)

            self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)])
            predict = self.model.predict(val_X)
            score = f1_score(val_y, predict, average="micro")
            scores[i_fold] = score

        score_mean = np.mean(scores)
        return score_mean

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        best_params = study.best_trial.params
        best_score = study.best_trial.value
        print(f"best params: {best_params}\nscore: {best_score}")
        self.history_plot(study)
        return best_params

    def history_plot(self, study):
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
