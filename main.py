import argparse
import os
import json
import datetime as dt
from restore_mnist.model import train_model
from restore_mnist.data_handling import build_split_images, load_data
from restore_mnist.inference import run_inference, compute_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="./data",
        help="Path to data (available at http://yann.lecun.com/exdb/mnist/)",
        type=str,
    )
    parser.add_argument(
        "--n_epochs",
        default=7,
        help="Number of training epochs",
        type=int,
    )

    parser.add_argument(
        "--n_candidates",
        default=300,
        help="Number of candidates for inference",
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="Path to output_directory",
        type=str,
    )
    args = parser.parse_args()

    print("Load data")
    train_images, test_images, train_original_labels, test_original_labels = load_data(
        args.data_path
    )

    print("Train model")
    model = train_model(
        train_images,
        test_images,
        train_original_labels,
        test_original_labels,
        n_epochs=args.n_epochs,
    )

    print("Run inference")
    split_images, labels = build_split_images(test_images)
    pairs = run_inference(split_images, model, args.n_candidates)

    accuracy = compute_accuracy(pairs, labels)
    print(f"Accuracy on test set: {100*accuracy:.3f}%")

    output = vars(args)
    output["accuracy"] = accuracy

    output_file = dt.datetime.now().strftime("output_%Y_%m_%d_%Hh_%Mm_%Ss.json")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f"{args.output_dir}/{output_file}", "w") as f:
        json.dump(output, f)
