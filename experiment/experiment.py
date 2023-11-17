import dataset
from dataset import Dataset


class Exp:
    def __init__(self, config) -> None:
        data: Dataset = getattr(dataset, config.data.name)()
        self.train = data.train
        self.test = data.test

    def run(self):
        print("run")
