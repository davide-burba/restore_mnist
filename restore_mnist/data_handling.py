from mnist import MNIST
import numpy as np
from tqdm import tqdm


def load_data(data_path):
    mndata = MNIST(data_path)

    train_images, _ = mndata.load_training()
    test_images, _ = mndata.load_testing()
    train_images = [np.array(img).reshape(28, 28) for img in tqdm(train_images)]
    test_images = [np.array(img).reshape(28, 28) for img in tqdm(test_images)]
    return train_images,test_images


def build_split_images(images):
    labels = []
    split_images = []
    for i, img in tqdm(enumerate(images)):
        left = img[:, :14]
        right = img[:, 14:]
        split_images.append(left)
        split_images.append(right)
        labels.append(f"left_{i}")
        labels.append(f"right_{i}")

    return split_images, labels
