import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
from wrapper import train_model_v2

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

default_mnist = tfds.load('mnist', as_supervised=True, data_dir=f'{os.getcwd()}/tensorflow_datasets/')
rotated_mnist = tfds.load('mnist_corrupted/rotate', as_supervised=True, data_dir=f'{os.getcwd()}/tensorflow_datasets/')
noisy_mnist = tfds.load('mnist_corrupted/impulse_noise', as_supervised=True, data_dir=f'{os.getcwd()}/tensorflow_datasets/')

def prepare_dataset(data):
    tmp_test, tmp_train= list(), list()
    for image, label in tqdm(tfds.as_numpy(data['test']), desc='Loading Part I of Data'):
        tmp_test.append(np.ndarray.flatten(image[0:28:2, 0:28:2]))

    for image, label in tqdm(tfds.as_numpy(data['train']), desc='Loading Part II of Data'):
        tmp_train.append(np.ndarray.flatten(image[0:28:2, 0:28:2]))

    return np.array(tmp_test).T, np.array(tmp_train).T


d_mnist_test, d_mnist_train = prepare_dataset(default_mnist)
r_mnist_test, r_mnist_train = prepare_dataset(rotated_mnist)
n_mnist_test, n_mnist_train = prepare_dataset(noisy_mnist)
print(d_mnist_test.shape, d_mnist_train.shape)

view1 = d_mnist_train
for ds, desc in [(r_mnist_train, 'ROTATED MNIST'), (n_mnist_train, 'NOISY MNIST')]:
    view2 = ds
    for sdim in [25, 15, 20]:
        try:
            train_model_v2(data=[view1, view2], batch_size=10000, shared_dim=sdim, hidden_dim=64,
                           pca_dim=50, epochs=10000, desc=desc)
        except Exception as e:
            print(f'EXCEPTION: {e}')

