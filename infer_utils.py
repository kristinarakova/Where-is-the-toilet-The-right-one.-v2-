from tqdm import tqdm
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import faiss
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


def build_index(data):
    n_obj, n_emb = data.shape
    index = faiss.IndexFlatL2(n_emb)
    index.add(data)
    return index


def search_index(data, k_neighbors=None):
    index = build_index(data)

    k_neighbors = k_neighbors or n_obj
    Dist, Ixes = index.search(data, k_neighbors)
    return Dist, Ixes


def compute_embedding(image, model, features):
    if "features" not in model._modules.keys():
        model = nn.Sequential(
            OrderedDict(
                [
                    ("features", nn.Sequential(*list(model.children())[:-2])),
                ]
            )
        )

    if features == "local":
        embedding = model.features(image)
        embedding = F.avg_pool2d(embedding, 7, 7).reshape(-1,)
    if features == "global":
        embedding = model(image)
    embedding = (
        embedding.cpu().detach().numpy().reshape(-1,)
    )
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def predict_opposite_crop_to_crop(model, features, dataloader, df_gt, device="cpu"):
    model.to(device)
    model.eval()
    embeddings = {}
    for inputs, label, names in tqdm(dataloader, total=len(dataloader)):
        for img, name in zip(inputs, names):
            img = img.to(device)
            embedding = compute_embedding(img, model, features)
            embeddings[name[0]] = embedding

    embs = np.array(list(embeddings.values()))
    files = np.array(list(embeddings.keys()))

    Dist, Ixes = search_index(embs, 2)

    df_result = pd.DataFrame([files, files[Ixes[:, 1]], Dist[:, 1]]).T
    df_result.columns = "path_crop", "path_crop_opposite", "dist"

    df_crop_to_crop = pd.merge(
        df_result,
        df_gt[["path_crop", "path_crop_opposite"]],
        on="path_crop",
        suffixes=("", "_gt"),
    )

    return df_crop_to_crop


def predict_original_to_crop(model, features, dataloader, df_gt, device="cpu"):
    model.eval()
    model.to(device)
    embeddings = {}
    for inputs, label, names in tqdm(dataloader, total=len(dataloader)):
        # add original images to index
        inp = inputs[1].to(device)
        
        embedding = compute_embedding(inp, model, features)
        embeddings[names[1][0]] = embedding

    embs = np.array(list(embeddings.values()))
    files = np.array(list(embeddings.keys()))
    index = build_index(embs)

    result = []
    for inputs, label, names in tqdm(dataloader, total=len(dataloader)):
        # compute embs for crop images and find the nearest
        inp = inputs[0].to(device)
        embedding = compute_embedding(inp, model, features)
        Dist, Ixes = index.search(embedding.reshape(1, -1), 1)
        result.append(
            {
                "path_crop": names[0][0],
                "path_orig": files[Ixes[0]][0],
                "dist": Dist[0][0],
            }
        )
    df_result = pd.DataFrame(result)
    df_crop_to_orig = pd.merge(
        df_result,
        df_gt[["path_crop", "path_orig"]],
        on="path_crop",
        suffixes=("", "_gt"),
    )
    return df_crop_to_orig


def predict_opposite_original_to_crop(
    df_crop_to_crop, df_crop_to_orig, df_crop_to_orig_opposite_gt
):
    df_crop_to_orig_opposite = pd.merge(
        df_crop_to_crop[["path_crop", "path_crop_opposite"]],
        df_crop_to_orig[["path_crop", "path_orig", "dist"]],
        left_on="path_crop_opposite",
        right_on="path_crop",
        suffixes=("", "_"),
    )[["path_crop", "path_crop_opposite", "path_orig", "dist"]]

    df_result = pd.merge(
        df_crop_to_orig_opposite,
        df_crop_to_orig_opposite_gt,
        on="path_crop",
        suffixes=("", "_gt"),
    )
    return df_result


def plot_match_pairs(df, num_pairs, target_name):
    for idx, row in df.iterrows():
        if idx == num_pairs:
            break
        img1 = Image.open(row["path_crop"]).convert("RGB")
        img2 = Image.open(row[target_name]).convert("RGB")

        fig, ax = plt.subplots(1, 2, figsize=(15, 4))

        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()
