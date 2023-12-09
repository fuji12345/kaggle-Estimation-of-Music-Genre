import model
import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def xgboost_config(trial):
    params_dict = {
        # "eta": trial.suggest_float("eta", 1e-5, 1.0, log=True),
        # "gamma": trial.suggest_float("gamma", 1e-8, 1e2, log=True),
        # "max_depth": trial.suggest_int("max_depth", 3, 10),
        # "min_child_weight": trial.suggest_int("min_child_weight", 1e-8, 8),
        # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        # "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
        # "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True),
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
    def __init__(self, model_name, X, y, cv, n_trials, config) -> None:
        self.config = config

        self.model_name = model_name
        self.X, self.y = X, y

        self.cv = cv
        self.n_trials = n_trials
        self.model = getattr(model, self.model_name)()

    def objective(self, trial):
        if not self.config.optuna.hold_out:
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
        else:
            train_val_X, test_X, train_val_y, test_y = train_test_split(self.X, self.y, test_size=0.2)
            train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size=0.2)

            self.model_params_dict = get_model_config(self.model_name, trial)

            self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)])
            predict = self.model.predict(test_X)
            score = f1_score(test_y, predict, average="micro")
            return score

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
