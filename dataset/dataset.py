import ssl

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from hydra.utils import to_absolute_path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

ssl._create_default_https_context = ssl._create_unverified_context


class MusicGenre:
    def __init__(self) -> None:
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        self.target_column = "genre"

        self.columns = [x for x in self.train.columns.tolist() if x != self.target_column]

        self.test_id_column = self.test["ID"]
        self.drop_id_and_type()

    def drop_id_and_type(self):
        self.train.drop(["ID", "type"], axis=1, inplace=True)
        self.test.drop(["ID", "type"], axis=1, inplace=True)

    def new_columns(self):
        self.train["energy_loudness"] = self.train["energy"] * self.train["loudness"]
        self.test["energy_loudness"] = self.test["energy"] * self.test["loudness"]

    def flatten_vector(self, vector):
        summed_vector = tf.reduce_sum(vector, axis=1)
        return tf.reshape(summed_vector, shape=(-1,))

    def song_vector(self):
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        vectors = embed(self.train["song_name"])
        test_vectors = embed(self.test["song_name"])
        flattened_vectors = self.flatten_vector(vectors)
        flattened_test_vectors = self.flatten_vector(test_vectors)
        self.train["song_vector"] = flattened_vectors.numpy()
        self.test["song_vector"] = flattened_test_vectors.numpy()
        self.train.drop(["song_name"], axis=1, inplace=True)
        self.test.drop(["song_name"], axis=1, inplace=True)

    def create_polynomial_features(self):
        features = ["instrumentalness", "danceability", "speechiness", "duration_ms", "valence"]
        pf = PolynomialFeatures(include_bias=False).fit(self.train[features])
        train_created_features = pd.DataFrame(pf.transform(self.train[features])[:, len(features) :])
        test_created_features = pd.DataFrame(pf.transform(self.test[features])[:, len(features) :])

        self.train = pd.concat([self.train, train_created_features], axis=1)
        self.test = pd.concat([self.test, test_created_features], axis=1)

    def label_encoding(self):
        self.le = LabelEncoder()
        self.train[self.target_column] = self.le.fit_transform(self.train[self.target_column])

    def inverse_label_encoding(self, predict):
        decode_predict = self.le.inverse_transform(predict)
        return decode_predict

    def preprocessing(self):
        self.create_polynomial_features()
        self.new_columns()
        self.song_vector()
