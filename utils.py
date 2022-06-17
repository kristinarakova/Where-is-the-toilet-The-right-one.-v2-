import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl


def load_image(image_path, ds_path="/home/korakova/kaggle"):
    image_path = os.path.join(ds_path, image_path)
    image = Image.open(image_path).convert("RGB")
    return image


def compute_accuracy(df, predict, target):
    return df.query(f"{predict}=={target}").shape[0] / df.shape[0]


mpl.rcParams["xtick.labelsize"] = "large"
mpl.rcParams["ytick.labelsize"] = "large"
mpl.rcParams["axes.labelsize"] = "x-large"
mpl.rcParams["legend.fontsize"] = "x-large"
mpl.rcParams["axes.titlesize"] = "xx-large"
mpl.rcParams["figure.titlesize"] = "xx-large"


def plot_train_results(train_result, title, yscale=False):
    plt.plot(train_result["train_loss"], label="train")
    plt.plot(train_result["valid_loss"], label="valid")
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.title(title)
    if yscale:
        plt.yscale("log")
    plt.legend()


def make_dir(path):
    if not os.path.exists(path):
        os.system(f"mkdir {path}")
