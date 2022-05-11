from mnist import MNIST
import numpy as np
from tqdm import tqdm


def load_data(data_path):
    mndata = MNIST(data_path)

    train_images, train_original_labels = mndata.load_training()
    test_images, test_original_labels = mndata.load_testing()
    train_images = [np.array(img).reshape(28, 28) for img in tqdm(train_images)]
    test_images = [np.array(img).reshape(28, 28) for img in tqdm(test_images)]
    return train_images,test_images, train_original_labels, test_original_labels


def build_split_images(images, original_labels=None):
    labels = []
    split_images = []
    for i, img in tqdm(enumerate(images)):
        left = img[:, :14]
        right = img[:, 14:]
        split_images.append(left)
        split_images.append(right)
        pattern = original_labels[i] if original_labels else ""
        labels.append(f"left_{pattern}_{i}")
        labels.append(f"right_{pattern}_{i}")

    return split_images, labels
