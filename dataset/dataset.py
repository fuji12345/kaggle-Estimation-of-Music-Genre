import ssl

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from hydra.utils import to_absolute_path
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ssl._create_default_https_context = ssl._create_unverified_context


class MusicGenre:
    def __init__(self, config) -> None:
        self.config = config

        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        self.target_column = "genre"

        self.columns = [x for x in self.train.columns.tolist() if x != self.target_column]
        self.use_columns = [x for x in self.columns if x not in ["ID", "type"]]

        self.categorical_columns = ["key", "mode"]

        self.contenious_columns = [x for x in self.use_columns if x not in self.categorical_columns]

        self.test_id_column = self.test["ID"]
        self.drop_id_and_type()

        classes = ["Dark Trap", "Emo", "Hiphop", "Pop", "Rap", "RnB", "Trap Metal", "Underground Rap"]
        class_weights = compute_class_weight("balanced", classes=classes, y=self.train[self.target_column])
        inverse_class_weights = 1.0 / class_weights
        self.scale_pos_weight = sum(inverse_class_weights) / len(inverse_class_weights)

    def drop_id_and_type(self):
        self.train.drop(["ID", "type"], axis=1, inplace=True)
        self.test.drop(["ID", "type"], axis=1, inplace=True)

    def new_columns(self):
        self.train["energy_loudness"] = self.train["energy"] * self.train["loudness"]
        self.test["energy_loudness"] = self.test["energy"] * self.test["loudness"]

    def diff_energy_loudness(self):
        self.train["diff_energy_loudness"] = self.train["energy"] - self.train["loudness"]
        self.test["diff_energy_loudness"] = self.test["energy"] - self.test["loudness"]

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
        new_columns_name = [f"A{x}" for x in range(15)]
        pf = PolynomialFeatures(include_bias=False).fit(self.train[features])
        train_created_features = pd.DataFrame(
            pf.transform(self.train[features])[:, len(features) :], columns=new_columns_name
        )
        test_created_features = pd.DataFrame(
            pf.transform(self.test[features])[:, len(features) :],
            columns=new_columns_name,
        )

        drop_columns = ["A0", "A5", "A9", "A12", "A14"]
        train_created_features.drop(drop_columns, axis=1, inplace=True)
        test_created_features.drop(drop_columns, axis=1, inplace=True)

        self.train = pd.concat([self.train, train_created_features], axis=1)
        self.test = pd.concat([self.test, test_created_features], axis=1)

    def create_features_PCA(self):
        contenious_train_data = self.train[self.contenious_columns]
        contenious_test_data = self.test[self.contenious_columns]
        scaler = StandardScaler()
        scaler = scaler.fit(contenious_train_data)
        pca_train_data_standardized = scaler.transform(contenious_train_data)
        pca_test_data_standardized = scaler.transform(contenious_test_data)

        pca = PCA(n_components=2)
        pca = pca.fit(pca_train_data_standardized)
        train_principal_components = pca.transform(pca_train_data_standardized)
        test_principal_components = pca.transform(pca_test_data_standardized)
        train_new_features = pd.DataFrame(data=train_principal_components, columns=["PC1", "PC2"])
        test_new_features = pd.DataFrame(data=test_principal_components, columns=["PC1", "PC2"])

        self.train = pd.concat([self.train, train_new_features], axis=1)
        self.test = pd.concat([self.test, test_new_features], axis=1)

    def columns_per_time_signature(self):
        train_duration_ms_per_time_signature = pd.DataFrame(
            self.train["duration_ms"] / self.train["time_signature"], columns=["duration_ms_per_time_signature"]
        )
        test_duration_ms_per_time_signature = pd.DataFrame(
            self.test["duration_ms"] / self.test["time_signature"], columns=["duration_ms_per_time_signature"]
        )

        self.train = pd.concat([self.train, train_duration_ms_per_time_signature], axis=1)
        self.test = pd.concat([self.test, test_duration_ms_per_time_signature], axis=1)

    def label_encoding(self):
        self.le = LabelEncoder()
        self.train[self.target_column] = self.le.fit_transform(self.train[self.target_column])

    def inverse_label_encoding(self, predict):
        decode_predict = self.le.inverse_transform(predict)
        return decode_predict

    def preprocessing(self):
        if self.config.preprocessing.per_time:
            self.columns_per_time_signature()

        if self.config.preprocessing.pca:
            self.create_features_PCA()

        if self.config.preprocessing.pf:
            self.create_polynomial_features()

        if self.config.preprocessing.new_col:
            self.new_columns()

        if self.config.preprocessing.diff_el:
            self.diff_energy_loudness()

        if self.config.preprocessing.song_vector:
            self.song_vector()
        else:
            self.train.drop(["song_name"], axis=1, inplace=True)
            self.test.drop(["song_name"], axis=1, inplace=True)
