import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class MusicGenre:
    def __init__(self) -> None:
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        self.columns = self.train.columns.tolist()
        self.target_column = "genre"

        self.test_id_column = self.test["ID"]
        self.drop_id_and_type()

    def drop_id_and_type(self):
        self.train.drop(["ID", "type"], axis=1, inplace=True)
        self.test.drop(["ID", "type"], axis=1, inplace=True)

    def label_encoding(self):
        self.le = LabelEncoder()
        self.train[self.target_column] = self.le.fit_transform(self.train[self.target_column])

    def inverse_label_encoding(self, predict):
        decode_predict = self.le.inverse_transform(predict)
        return decode_predict

    def preprocessing(self):
        self.train.drop("song_name", axis=1, inplace=True)
        self.test.drop("song_name", axis=1, inplace=True)
