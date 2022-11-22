import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Function to calculate the correlations of a model with TID-2013 or KADIK-10K
def calculate_correlation(model, dataset = 'tid'):
    # Load the database
    if dataset == 'tid':
        dataset_path = '/lustre/ific.uv.es/ml/uv075/Databases/IQA/TID/TID2013'
        data = pd.read_csv(dataset_path + '/image_pairs_mos.csv', index_col = 0)

        def generator():
            for _, row in data.iterrows():
                img = cv2.imread(dataset_path + '/reference_images/' + row.Reference)
                dist_img = cv2.imread(dataset_path + '/distorted_images/' + row.Distorted)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
                img = img/255.0
                dist_img = dist_img/255.0
                yield img, dist_img, row.MOS

    elif dataset == 'kadik':
        dataset_path = '/lustre/ific.uv.es/ml/uv075/Databases/IQA/KADIK10K'
        data = pd.read_csv(dataset_path + '/dmos.csv')

        def generator():
            for _, row in data.iterrows():
                img = cv2.imread(dataset_path + '/images/' + row.ref_img)
                dist_img = cv2.imread(dataset_path + '/images/' + row.dist_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
                img = img/255.0
                dist_img = dist_img/255.0
                yield img, dist_img, row.dmos

    else:
        raise ValueError('Dataset not supported')
    
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                             tf.TensorSpec(shape = (384, 512, 3), dtype=tf.float32),
                                             tf.TensorSpec(shape = (384, 512, 3), dtype=tf.float32),
                                             tf.TensorSpec(shape = (), dtype=tf.float32)))

    # Get every layer of the model
    layers = [layer for layer in model.layers]
    distances = {i:[] for i in range(len(layers)+1)}
    mos_data = []

    # Calculate the distances between images and distorted images using the outputs of every layer of the model
    def calculate_distance(images, dist_images):
        l2 = (np.array(images)-np.array(dist_images))**2
        l2 = tf.reduce_sum(l2, axis = [1,2,3])
        l2 = tf.sqrt(l2)
        return l2

    for _, (imgs, dists, mos) in enumerate(dataset.batch(32)):
        mos_data.extend(mos)
        layer_outputs_img = imgs
        layer_outputs_dists = dists
        distances[0].extend(calculate_distance(layer_outputs_img, layer_outputs_dists))

        for i in range(len(layers)):
            layer_outputs_img = layers[i](layer_outputs_img)
            layer_outputs_dists = layers[i](layer_outputs_dists)
            distances[i+1].extend(calculate_distance(layer_outputs_img, layer_outputs_dists))

    pearson_correlations, spearman_correlations = [], []

    # Calculate the correlation at every layer of the model
    for i in range(len(layers)+1):
        pearson = stats.pearsonr(mos_data, distances[i])[0]
        spearman = stats.spearmanr(mos_data, distances[i])[0]
        pearson_correlations.append(pearson)
        spearman_correlations.append(spearman)

    return pearson_correlations, spearman_correlations


# Function to plot the correlations at every layer of PerceptNet-Autoencoder
def plot_correlation_perceptnet_autoencoder(pearson_correlation, spearman_correlation, dataset = 'tid', save_path_fig = './'):
    pearson_correlation, spearman_correlation = np.array(pearson_correlation), np.array(spearman_correlation)

    # To have always positive correlations for a better visualization (is easier to understand that up is good and low is bad than viceversa)
    if pearson_correlation[0] < 0:
        pearson_correlation = -pearson_correlation
    if spearman_correlation[0] < 0:
        spearman_correlation = -spearman_correlation

    # Get the corresponding database
    if dataset == 'tid':
        dataset = 'TID-2013'
    elif dataset == 'kadik':
        dataset = 'KADIK-10K'
    else:
        raise ValueError('Dataset not supported')

    # Define model layers
    layers = ['input', 'gdn1', 'conv1', 'max_pool1', 'gdn2', 'conv2', 'max_pool2', 'gdn3', 'conv3', 'gdn4', 
              'inv_gdn1', 'conv4', 'inv_gdn2', 'up_sampl1', 'conv5', 'inv_gdn3', 'up_sampl2', 'conv6', 'inv_gdn4']

    # Plot
    plt.figure(figsize=(12,5), facecolor=(1, 1, 1))
    plt.plot(np.arange(19), pearson_correlation, '-o', label = f'{dataset} Pearson corr')
    plt.plot(np.arange(19), spearman_correlation, '-o', label = f'{dataset} Spearman corr')
    plt.axvspan(-0.5, 0.5, color='m', alpha=0.2, lw=0)
    plt.axvspan(0.5, 3.5, color='b', alpha=0.2, lw=0)
    plt.axvspan(3.5, 6.5, color='y', alpha=0.2, lw=0)
    plt.axvspan(6.5, 9.5, color='r', alpha=0.2, lw=0)
    plt.axvspan(9.5, 12.5, color='r', alpha=0.1, lw=0)
    plt.axvspan(12.5, 15.5, color='y', alpha=0.1, lw=0)
    plt.axvspan(15.5, 18.5, color='b', alpha=0.1, lw=0)
    plt.axvline(x = 9.5, color='k')
    plt.xticks(np.arange(19), layers, rotation = 45)
    plt.title(f'Perceptnet autoencoder {dataset} correlations')
    plt.xlabel('Layer')
    plt.ylabel('Correlation')
    #plt.ylim([0.5,1])
    plt.xlim([-0.5,18.5])
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.savefig(save_path_fig + 'perceptnet_autoencoder_correlations.png')