import pathlib
import csv

import torch
from torchvision.datasets import GTSRB
from torchvision.datasets.folder import make_dataset

class traffic(GTSRB):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(GTSRB, self).__init__(root, transform, target_transform)
        self._split = split
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Final_Training" if self._split == "train" else "Final_Test/Images")
        )
        if self._split == "train":
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform