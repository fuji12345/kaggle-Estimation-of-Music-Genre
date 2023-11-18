import pandas as pd
from hydra.utils import to_absolute_path

import dataset
import model


class Exp:
    def __init__(self, config) -> None:
        self.data = getattr(dataset, config.data.name)()
        self.data.label_encoding()
        self.data.preprocessing()

        self.test_id_column = self.data.test_id_column

        self.train = self.data.train
        self.test = self.data.test

        self.columns = self.data.columns
        self.target_column = self.data.target_column

        self.model = getattr(model, config.model.name)()

    def run(self):
        train_X, train_y = self.get_x_y(self.train)

        self.model.fit(train_X, train_y)
        predict = self.model.predict(self.test)

        self.make_output_file(predict)

    def get_x_y(self, data: pd.DataFrame):
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        return X, y

    def make_output_file(self, predict, output_path="outputs/submmition.csv"):
        decode_predict = self.data.inverse_label_encoding(predict)
        decode_predict = pd.DataFrame(decode_predict, columns=[self.target_column])
        output_df = pd.concat([self.data.test_id_column, decode_predict], axis=1)
        output_df.to_csv(to_absolute_path(output_path), index=False)
