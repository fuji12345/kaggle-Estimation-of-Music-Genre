from pathlib import Path

import matplotlib.pyplot as plt
import model
import numpy as np
import optuna
from hydra.utils import to_absolute_path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def xgboost_config(trial):
    params_dict = {
        # "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        # "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        # "max_depth": trial.suggest_int("max_depth", 3, 9),
        # "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        # "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        # "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        #
        "eta": trial.suggest_float("eta", 0.0001, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 1.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 100),
    }
    return params_dict


def get_model_config(model_name, trial):
    if model_name == "XGBoost":
        return xgboost_config(trial)
    else:
        return ValueError()


class Optuna:
    def __init__(self, config, X, y, i_fold=None) -> None:
        self.config = config
        self.i_fold = i_fold
        self.X, self.y = X, y
        self.model = getattr(model, self.config.model.name)(seed=self.config.seed, early_stopping_rounds=50)

    def objective(self, trial):
        if not self.config.optuna.hold_out:
            skf = StratifiedKFold(n_splits=self.config.optuna.cv, shuffle=True, random_state=self.config.seed)
            scores = np.zeros(self.config.optuna.cv)
            for i_fold, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
                train_X, train_y = self.X.iloc[train_index], self.y.iloc[train_index]
                val_X, val_y = self.X.iloc[val_index], self.y.iloc[val_index]

                self.model_params_dict = get_model_config(self.config.model.name, trial)

                self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
                predict = self.model.predict(val_X)
                score = f1_score(val_y, predict, average="micro")
                scores[i_fold] = score

            score_mean = np.mean(scores)
            return score_mean
        else:
            train_val_X, test_X, train_val_y, test_y = train_test_split(self.X, self.y, test_size=0.2)
            train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size=0.2)

            self.model_params_dict = get_model_config(self.config.model.name, trial)

            self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
            predict = self.model.predict(test_X)
            score = f1_score(test_y, predict, average="micro")
            return score

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.optuna.n_trials, n_jobs=1)

        best_params = study.best_trial.params
        best_score = study.best_trial.value
        print(f"best params: {best_params}\nscore: {best_score}")
        self.history_plot(study)
        return best_params

    def history_plot(self, study):
        path = Path(to_absolute_path(f"outputs/optuna_history/result{self.i_fold}.png"))
        path.parent.mkdir(exist_ok=True, parents=True)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(to_absolute_path(path))
