import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transforms import TRAIN, INFER
from sklearn.model_selection import train_test_split
from utils import load_image


class PairDataset(Dataset):
    def __init__(self, file_splits, transform, dataset_path):
        self.inputs = file_splits
        self.transform = transform
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        image_name1 = self.inputs["path1"].values[index]
        image_name2 = self.inputs["path2"].values[index]
        label = self.inputs["target"].values[index]

        image1 = load_image(image_name1)
        image1 = self.transform(image1)

        image2 = load_image(image_name2)
        image2 = self.transform(image2)

        return (image1, image2), label, (image_name1, image_name2)


class ClassificationDataset(Dataset):
    def __init__(self, file_splits, transform, dataset_path, classes):
        self.inputs = file_splits
        self.transform = transform
        self.dataset_path = dataset_path
        self.classes = classes

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        image_name = self.inputs["path"].values[index]
        label = self.inputs["pair_id"].values[index]

        image = load_image(image_name)
        image = self.transform(image)
        label = self.classes.index(label)

        return (image,), label, (image_name,)


class SubmissionDataset(Dataset):
    def __init__(self, file_splits, transform, dataset_path):
        self.inputs = file_splits
        self.transform = transform
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        image_name = self.inputs["path"].values[index]
        label = self.inputs["is_query"].values[index]
        image = load_image(image_name)
        image = self.transform(image)

        return image, label, image_name


def prepare_data(data, task_type, is_test):
    if task_type == "crop_crop":
        positive_pairs = []
        for idx, df in data.groupby("pair_id"):
            positive_pairs.append(
                {
                    "path1": df.path_crop.values[0],
                    "path2": df.path_crop.values[1],
                    "target": 1,
                }
            )
        df_pos = pd.DataFrame(positive_pairs)

    if task_type == "crop_orig":
        df_pos = data[["path_crop", "path_orig"]]
        df_pos["target"] = 1
        df_pos.columns = "path1", "path2", "target"

    files = list(df_pos.path1.values)
    files.extend(list(df_pos.path2.values))

    num_negative = df_pos.shape[0] * 2
    negative_pairs = []
    for i in range(num_negative):
        file1, file2 = np.random.choice(files, size=2)
        negative_pairs.append({"path1": file1, "path2": file2, "target": 0})

    df_neg = pd.DataFrame(negative_pairs)
    df = pd.concat([df_pos, df_neg])
    df = df.drop_duplicates(["path1", "path2"])
    if is_test:
        return df.query("target==1")
    else:
        return df
    return df


def prepare_data_classification(train):
    columns = "pair_id", "path"
    data_crop = train[["pair_id", "path_crop"]]
    data_crop.columns = columns

    data_orig = train[["pair_id", "path_orig"]]
    data_orig.columns = columns

    data = pd.concat([data_crop, data_orig])

    train_data, valid_data = train_test_split(
        data, train_size=0.7, random_state=42, stratify=data.pair_id.values
    )

    return train_data, valid_data


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test


def split_train_data(train):
    train_pair_id, valid_pair_id = train_test_split(
        train.pair_id.unique(), train_size=0.7, random_state=42
    )
    train_ = train.query("pair_id in @train_pair_id")
    valid_ = train.query("pair_id in @valid_pair_id")
    return train_, valid_


def make_crop_to_opposite_original_df(data):
    df_gt = pd.concat(
        [
            pd.merge(
                data.query("gender=='f'"),
                data.query("gender=='m'"),
                on="pair_id",
                suffixes=("", "_opposite"),
            ),
            pd.merge(
                data.query("gender=='m'"),
                data.query("gender=='f'"),
                on="pair_id",
                suffixes=("", "_opposite"),
            ),
        ]
    )
    return df_gt


def get_crop_to_opposite_original_df():
    train, test = load_data()
    train_, valid_ = split_train_data(train)
    return make_crop_to_opposite_original_df(valid_)


def make_dataloaders(
    task_type,
    batch_size,
    num_workers=10,
    data_path="/home/korakova/kaggle/",
):
    train, test = load_data()
    train_, valid_ = split_train_data(train)

    if task_type == "classification":
        classes = list(train_.pair_id.unique())
        num_classes = len(classes)

        train_data, valid_data = prepare_data_classification(train_)

        train_dataset = ClassificationDataset(train_data, TRAIN, data_path, classes)
        valid_dataset = ClassificationDataset(valid_data, INFER, data_path, classes)

        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size, shuffle=True, num_workers=num_workers
            ),
            "valid": DataLoader(
                valid_dataset, batch_size, shuffle=False, num_workers=num_workers
            ),
        }
        return dataloaders, num_classes

    elif task_type in ["crop_crop", "crop_orig"]:
        train_data = prepare_data(train_, task_type, is_test=False)
        valid_data = prepare_data(valid_, task_type, is_test=False)

        train_dataset = PairDataset(train_data, TRAIN, data_path)
        valid_dataset = PairDataset(valid_data, INFER, data_path)

        test_data = prepare_data(valid_, task_type, is_test=True)
        test_dataset = PairDataset(test_data, INFER, data_path)

        full_data = prepare_data(train, task_type, is_test=True)
        full_dataset = PairDataset(full_data, INFER, data_path)

        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size, shuffle=True, num_workers=num_workers
            ),
            "valid": DataLoader(
                valid_dataset, batch_size, shuffle=False, num_workers=num_workers
            ),
            "test": DataLoader(test_dataset, 1, shuffle=False, num_workers=num_workers),
            "full": DataLoader(full_dataset, 1, shuffle=False, num_workers=num_workers),
        }
        return dataloaders
    else:
        raise Exception(f"Not implemented task type {task_type}")


def make_submission_dataloader(
    batch_size=1,
    num_workers=10,
    data_path="/home/korakova/kaggle/",
):
    train, test = load_data()

    query_dataset = SubmissionDataset(test.query("is_query==1"), INFER, data_path)
    non_query_dataset = SubmissionDataset(test.query("is_query==0"), INFER, data_path)
    dataloaders = {
        "query": DataLoader(
            query_dataset, batch_size, shuffle=False, num_workers=num_workers
        ),
        "non_query": DataLoader(
            non_query_dataset, batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return dataloaders
