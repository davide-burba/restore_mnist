# Restore mnist

This repository aims at solving the following task:

_Imagine the digits in the test set of the MNIST dataset
(http://yann.lecun.com/exdb/mnist/) got cut in half vertically and shuffled
around. Implement a way to restore the original test set from the two halves,
whilst maximising the overall matching accuracy._

The following steps are made:
1. Split all the image in halfs
2. Train a binary classifier to predict if an image is made by two matching halfs or not
3. Take the half images. Use a rule of thumb to distinguish lefts and rights (based on the sum of left/right pixels)
4. For each left image, select a set of candidates for its right based on the distance of the touching edges
5. Take the candidate with the highest probability of being a match
6. Compute matches accuracies

The training architecture and part of the code was copied from [here](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/).

To execute the code: 
```
poetry install
poetry run python main.py  [-h] [--data_path DATA_PATH] [--n_epochs N_EPOCHS] [--n_candidates N_CANDIDATES] [--output_dir OUTPUT_DIR]
```

## Results

With the default parameters, we get an accuracy of 45.6%.
Note that, at inference time, this is framed as a classification problem with as many classes as the number images (10k). 
Therefore, any accuracy above 0.01% is better than a random guess.
If we frame the problem as reconstructing images with matching number labels for left and right side (i.e. 10 classes), than we get an accuracy of ~100%.

## Possible improvements

- training and inference distribution is different. Training could be improved by building the wrong images according to edge distances.
- the model hyperparameters could be optimized
- the rule to make sure that the number of left and the number of right halfs is the same should be reviewed, e.g. by constructing a ranking based on the ratio of the sum of the left/right edges
- the training data in the binary task is balanced by downsampling the majority class to the same size of the minority one. It could be worth to try different ratios.
- code could be made more efficient (especially at inference time), better documented, and refactored
