import os
import urllib
import urllib.request
from zipfile import ZipFile

###############################################
# Data preparation
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip' 
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE} ...')
    urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images: 
    zip_images.extractall(FOLDER)

print('Done!')

###############################################
# Data loading
import numpy as np
import cv2
import os

# Load dataset
def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))    # scan all directories & create a list of labels
    # create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # For each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            # append to the lists
            X.append(image)
            y.append(label)

    # Convert data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path) 
    X_test, y_test =load_mnist_dataset('test', path)
    return X, y, X_test, y_test

# create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')