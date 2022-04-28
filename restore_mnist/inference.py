from scipy.spatial import distance_matrix
import numpy as np
from tqdm import tqdm

from restore_mnist.model import prep_pixels


def distinguish_left_right(split_images):
    sides = {
        "right": [],
        "left": [],
    }
    for i, img in enumerate(split_images):
        if img[:, 0].sum() > img[:, -1].sum():
            sides["right"].append(i)
        else:
            sides["left"].append(i)
    return sides


def find_pair_candidates(dist, sides):
    pair_candidates = {}
    for i in tqdm(range(dist.shape[0])):
        # j_candidates = np.argpartition(dist[i], n)[:n]
        j_candidates = np.argsort(dist[i])
        pair_candidates[sides["left"][i]] = [
            sides["right"][int(j)] for j in j_candidates
        ]
    return pair_candidates


def run_inference(split_images, model, n_candidates=300):
    sides = distinguish_left_right(split_images)
    # build distance matrix
    x_left = np.concatenate(
        [split_images[i][:, -1].reshape(1, -1) for i in sides["left"]], axis=0
    )
    x_right = np.concatenate(
        [split_images[i][:, 0].reshape(1, -1) for i in sides["right"]], axis=0
    )
    dist = distance_matrix(x_left, x_right)

    # find candidates
    pair_candidates = find_pair_candidates(dist, sides)

    # take best candidates
    pairs = []
    taken = set()
    for i, l in tqdm(pair_candidates.items()):

        l_sel = [j for j in l if j not in taken][:n_candidates]

        candidates = np.concatenate(
            [
                np.concatenate([split_images[i], split_images[j]], axis=1).reshape(
                    1, 28, 28
                )
                for j in l_sel
            ],
            0,
        )
        candidates = prep_pixels(candidates)
        pred = model.predict(candidates)[:, 1]
        j = l_sel[np.argmax(pred)]
        pairs.append((i, j))
        taken.add(j)

    return pairs


def compute_accuracy(pairs, labels):
    is_correct = []
    for i, j in pairs:
        if labels[i].split("_")[1] == labels[j].split("_")[1]:
            is_correct.append(1)
        else:
            is_correct.append(0)

    return np.mean(is_correct).item()
