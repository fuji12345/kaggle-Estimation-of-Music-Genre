from typing import Dict

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import dataset
import model
from model import XGBoost

from .custom_optuna import Optuna


class Exp:
    def __init__(self, config) -> None:
        self.model_name = config.model.name
        self.n_splits = config.n_splits

        self.use_optuna = config.optuna.use_optuna
        self.optuna_cv = config.optuna.cv
        self.optuna_n_trials = config.optuna.n_trials

        self.is_average_voting = config.voting.average

        self.data = getattr(dataset, config.data.name)()
        self.data.label_encoding()
        self.data.preprocessing()

        self.test_id_column = self.data.test_id_column

        self.train = self.data.train
        self.test = self.data.test

        self.columns = self.data.columns
        self.target_column = self.data.target_column

        self.models_dict: Dict[int:XGBoost] = {}
        self.test_predicts: Dict[int : np.array] = {}
        self.test_predict_probas: Dict[int : np.array] = {}

    def get_x_y(self, data: pd.DataFrame):
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        return X, y

    def majority_voting_of_predict(self):
        predict_concat = np.zeros((self.test.shape[0], self.n_splits))
        for i_fold, predict in self.test_predicts.items():
            predict_concat[:, i_fold] = predict

        predict = stats.mode(predict_concat, axis=1)[0].flatten().astype(np.int64)
        return predict

    def average_voting_of_predict_proba(self):
        predict_probas_list = [x for x in self.test_predict_probas.values()]
        predict_proba_three_dimension = np.array(predict_probas_list)
        predict_proba_mean = np.mean(predict_proba_three_dimension, axis=0)
        predict = np.argmax(predict_proba_mean, axis=1)
        return predict

    def make_output_file(self, predict, output_path="outputs/submmition.csv"):
        decode_predict = self.data.inverse_label_encoding(predict)
        decode_predict = pd.DataFrame(decode_predict, columns=[self.target_column])
        output_df = pd.concat([self.data.test_id_column, decode_predict], axis=1)
        output_df.to_csv(to_absolute_path(output_path), index=False)

    def each_fold(self, i_fold, train_data, val_data, best_params):
        train_X, train_y = self.get_x_y(train_data)
        current_model = getattr(model, self.model_name)()
        if self.use_optuna:
            current_model.set_params(best_params)
        current_model.fit(train_X, train_y)

        train_predict = current_model.predict(train_X)
        train_score = f1_score(train_y, train_predict, average="micro")

        val_X, val_y = self.get_x_y(val_data)
        val_predict = current_model.predict(val_X)
        val_score = f1_score(val_y, val_predict, average="micro")

        if self.is_average_voting:
            test_predict_proba = current_model.predict_proba(self.test)
            self.test_predict_probas[i_fold] = test_predict_proba
        else:
            test_predict = current_model.predict(self.test)
            self.test_predicts[i_fold] = test_predict

        self.models_dict[i_fold] = current_model

        print(f"cv {i_fold}, train_score: {train_score}, val_score: {val_score}")

    def run(self):
        train_X, train_y = self.get_x_y(self.train)

        best_params = None
        if self.use_optuna:
            best_params: Dict = Optuna(self.model_name, train_X, train_y, self.optuna_cv, self.optuna_n_trials).run()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        for i_fold, (train_index, val_index) in enumerate(skf.split(train_X, train_y)):
            train_data, val_data = self.train.iloc[train_index], self.train.iloc[val_index]
            self.each_fold(i_fold, train_data, val_data, best_params)

        if self.is_average_voting:
            predict = self.average_voting_of_predict_proba()
            print("average voting")
        else:
            predict = self.majority_voting_of_predict()
            print("majority voting")

        self.make_output_file(predict)
