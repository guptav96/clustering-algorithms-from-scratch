import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(0)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def random_pick(digits_raw, digit):
    img_array = digits_raw[digits_raw[1] == digit].sample()
    img_array = img_array.iloc[0, 2:].values.reshape(28,28)
    return img_array 

def visualize_mnist_digits_raw(filename):
    # read the raw data file
    digits_raw = pd.read_csv(filename, header = None)
    # pick a data point for each class
    img_arrays = [random_pick(digits_raw, digit) for digit in range(10)]
    # plot digits
    _, ax = plt.subplots(2,5)
    for row in range(2):
        for col in range(5):
            img = ax[row,col].imshow(img_arrays[row*5 + col], cmap='gray')
            ax[row,col].axis('off')
            ax[row,col].set_title(f'Label: {row * 5  + col}')
    plt.suptitle('Visualizing MNIST Digits')
    plt.show()

def visualize_tsne_embeddings(filename):
    # read embeddings file
    digits_embedding = pd.read_csv(filename, header = None)
    # total number of samples
    N = len(digits_embedding)
    # randomly sample 1000 data points
    random_samples = np.random.randint(0, N, size=1000)
    selected_embeddings = digits_embedding[digits_embedding.index.isin(random_samples)]
    # plot the embeddings in 2D plot
    for idx, df_grouped in selected_embeddings.groupby(1):
        plt.scatter(df_grouped[2], df_grouped[3], c = colors[idx], label = f'Digit {idx}')
    plt.title('Visualize MNIST Digits in 2D')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    visualize_mnist_digits_raw('digits-raw.csv')
    visualize_tsne_embeddings('digits-embedding.csv')
