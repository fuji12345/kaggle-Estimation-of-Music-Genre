import pandas as pd
from hydra.utils import to_absolute_path


class Dataset:
    def __init__(self) -> None:
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))
