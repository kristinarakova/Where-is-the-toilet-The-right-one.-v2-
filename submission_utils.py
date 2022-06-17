from tqdm import tqdm
import numpy as np
import pandas as pd

from infer_utils import (
    build_index,
    search_index,
    compute_embedding,
)
from dataset import load_data


def predict_opposite_crop_to_crop_submit(model, features, dataloader):
    model.eval()
    embeddings = {}
    for inputs, label, names in tqdm(dataloader, total=len(dataloader)):
        embedding = compute_embedding(inputs, model, features)
        embeddings[names[0]] = embedding

    embs = np.array(list(embeddings.values()))
    files = np.array(list(embeddings.keys()))

    Dist, Ixes = search_index(embs, 2)

    df_result = pd.DataFrame([files, files[Ixes[:, 1]], Dist[:, 1]]).T
    df_result.columns = "path_crop", "path_crop_opposite", "dist"
    return df_result


def predict_original_to_crop_submit(model, features, dataloaders):
    model.eval()

    embeddings = {}
    for inputs, label, names in tqdm(
        dataloaders["non_query"], total=len(dataloaders["non_query"])
    ):
        # add original images to index
        embedding = compute_embedding(inputs, model, features)
        embeddings[names[0]] = embedding

    embs = np.array(list(embeddings.values()))
    files = np.array(list(embeddings.keys()))
    index = build_index(embs)

    result = []
    for inputs, label, names in tqdm(
        dataloaders["query"], total=len(dataloaders["query"])
    ):
        # compute embs for crop images and find the nearest
        embedding = compute_embedding(inputs, model, features)
        Dist, Ixes = index.search(embedding.reshape(1, -1), 1)
        result.append(
            {
                "path_crop": names[0],
                "path_orig": files[Ixes[0]][0],
                "dist": Dist[0][0],
            }
        )
    return pd.DataFrame(result)


def make_submission_file(df_crop_to_crop, df_crop_to_orig):
    train, test = load_data()
    df_crop_to_opposite_orig = pd.merge(
        df_crop_to_crop[["path_crop", "path_crop_opposite"]],
        df_crop_to_orig,
        left_on="path_crop_opposite",
        right_on="path_crop",
        suffixes=("", "_opposite"),
    )

    df1 = pd.merge(
        test.reset_index(),
        df_crop_to_opposite_orig[["path_crop", "path_orig", "dist"]],
        left_on="path",
        right_on="path_crop",
    )

    df2 = pd.merge(
        df1, test.reset_index()[["path", "index"]], left_on="path_orig", right_on="path"
    )
    res = df2.sort_values("index_x")

    submission = res[["index_x", "dist", "index_y"]]
    submission.loc[:, "Iteration"] = None
    submission.loc[:, "Rank"] = 0
    submission.loc[:, "RunId"] = None
    submission.columns = (
        "QueryId",
        "Similarity",
        "DocumentNumber",
        "Iteration",
        "Rank",
        "RunId",
    )
    submission = submission[
        ["QueryId", "Iteration", "DocumentNumber", "Rank", "Similarity", "RunId"]
    ]
    return submission
