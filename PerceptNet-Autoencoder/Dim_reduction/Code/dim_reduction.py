import tensorflow as tf
import pandas as pd
import cv2
from scipy import stats
import pickle
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback

from utils import *

# Wandb configuration parameters
config = {
    "batch_size": 128,
    "learning_rate": 0.0005,
    "seed":123,
    "epochs":80,
    "batch_num_save":70,
}

wandb.init(project="DL_and_Perception",
           name="Dim_reduction",
           config=config)
config = wandb.config

# Load TID-2013 data for calculate the correlations during the training
tid_path_2013 = '/lustre/ific.uv.es/ml/uv075/Databases/IQA/TID/TID2013'
data_tid_2013 = pd.read_csv(tid_path_2013 + '/image_pairs_mos.csv', index_col = 0)

def train_gen_tid2013():
    for i, row in data_tid_2013.iterrows():
        img = cv2.imread(tid_path_2013 + '/reference_images/' + row.Reference)
        dist_img = cv2.imread(tid_path_2013 + '/distorted_images/' + row.Distorted)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        dist_img = dist_img/255.0
        yield img, dist_img, row.MOS

tid2013_dataset = tf.data.Dataset.from_generator(train_gen_tid2013,
                                                 output_signature=(
                                                 tf.TensorSpec(shape = (384, 512, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape = (384, 512, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape = (), dtype=tf.float32)))

print(tid2013_dataset)

# Load ImageNet data to train the autoencoder
path = '/lustre/ific.uv.es/ml/uv075/Databases/imagenet_images'

train_ds = tf.keras.utils.image_dataset_from_directory(path,
                                                       labels = None,
                                                       seed = config.seed,
                                                       image_size = (256, 256),
                                                       batch_size = config.batch_size,
                                                       crop_to_aspect_ratio = True,
                                                       shuffle = True,
                                                       validation_split = 0.2,
                                                       subset = 'training')
val_ds = tf.keras.utils.image_dataset_from_directory(path,
                                                     labels = None,
                                                     seed = config.seed,
                                                     image_size = (256, 256),
                                                     batch_size = config.batch_size,
                                                     crop_to_aspect_ratio = True,
                                                     shuffle = True,
                                                     validation_split = 0.2,
                                                     subset = 'validation')

def normalize(image):
    return tf.cast(image/255., tf.float32)

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

print(train_ds, val_ds)

# Functions to calculate the correlation with TID-2013 at the bottleneck and at the end of the autoencoder. Also a function to calculate the autoencoder validation rmse
@tf.function
def tid_step(img, dist, perceptnet):
    pred_img = perceptnet(img)
    pred_dist = perceptnet(dist)
    l2 = (pred_img-pred_dist)**2
    l2 = tf.reduce_sum(l2, axis = [1,2,3])
    l2 = tf.sqrt(l2)
    return l2

@tf.function
def tid_step_end_distance(img, dist, model):
    pred_img = model(img)
    pred_dist = model(dist)
    l2 = (pred_img-pred_dist)**2
    l2 = tf.reduce_sum(l2, axis = [1,2,3])
    l2 = tf.sqrt(l2)
    return l2

@tf.function
def test_step(images, model):
    reconstucted_images = model(images)
    l2 = (reconstucted_images-images)**2
    l2 = tf.reduce_sum(l2, axis = [1,2,3])/(images.shape[1]*images.shape[2]*images.shape[3])
    loss = tf.sqrt(l2)
    loss = tf.reduce_mean(loss)
    return loss


# Callback to calculate every X batches the correlation with TID-2013 at the middle and at the end of the encoder. Also saves the training/validation loss
class SaveBatchLoss(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, batch_num = 30):
        self.batch_num = batch_num
        self.val_ds = val_ds
        self.batch_loss = []
        self.batch_loss_val = []
        self.pearson_correlation = []
        self.pearson_end_correlation = []

    def on_train_begin(self, logs = None):
        self.perceptnet = tf.keras.Sequential(self.model.layers[:9])
        
        moses, distances = [], []
        for img, dist_img, mos in tid2013_dataset.batch(16):
            moses.extend(mos)
            l2 = tid_step(img, dist_img, self.perceptnet)
            distances.extend(l2)
        self.pearson_correlation.append(stats.pearsonr(moses, distances)[0])

        
    def on_train_batch_end(self, batch, logs = None):
        self.batch_loss.append(logs['reconstruction_loss'])

        if batch % self.batch_num == 0:
            self.perceptnet = tf.keras.Sequential(self.model.layers[:9])
        
            moses, distances, distances_end = [], [], []
            for img, dist_img, mos in tid2013_dataset.batch(16):
                moses.extend(mos)

                l2 = tid_step(img, dist_img, self.perceptnet)
                distances.extend(l2)

                l2_end = tid_step_end_distance(img, dist_img, self.model)
                distances_end.extend(l2_end)

            self.pearson_correlation.append(stats.pearsonr(moses, distances)[0])
            self.pearson_end_correlation.append(stats.pearsonr(moses, distances_end)[0])

            val_loss = []
            for _, val_images in enumerate(self.val_ds):
                val_loss_value = test_step(val_images, self.model)
                val_loss.append(val_loss_value)
            
            self.batch_loss_val.append(tf.reduce_mean(val_loss))

# Build the autoendoder model
print('Building the model')
model = PerceptNetAutoEncoder(kernel_initializer = 'ones', 
                              gdn_kernel_size = 1)

model.compile(optimizer = Adam(learning_rate = config.learning_rate),
              loss = None)

model.build(input_shape = (None, 384, 512, 3))
print(model.summary())

# Initialize the callbacks
callback_history = SaveBatchLoss(val_ds = val_ds, 
                                 batch_num = config.batch_num_save)
checkpoint = tf.keras.callbacks.ModelCheckpoint(f'../models/dim_reduction_{config.seed}.h5', 
                                                monitor = "val_reconstruction_loss", 
                                                mode = 'min', 
                                                save_weights_only = True)
cb_wandb = WandbCallback(monitor="val_reconstruction_loss",
                         mode="min",
                         save_model=True,
                         save_weights_only=True)

# Train the autoencoder
history_t = model.fit(train_ds, epochs = config.epochs, callbacks = [callback_history, checkpoint, cb_wandb])

wandb.finish()

# Save training history
history = {'train_loss': callback_history.batch_loss,
           'val_loss': callback_history.batch_loss_val,
           'correlations_middle': callback_history.pearson_correlation,
           'correlations_end': callback_history.pearson_end_correlation}

results_history = open(f"../histories/dim_reduction_{config.seed}.pkl", "wb")
pickle.dump(history, results_history)
results_history.close()