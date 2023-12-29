#Copyright 2023 Google LLC

#Use of this source code is governed by an MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.

"""coco_data.py
Data pre-processing and loading pipeline for COCO-Stuff-27.
"""

import os
import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf

RESOLUTION = 512
AUTO = tf.data.AUTOTUNE

augmenter = keras.Sequential(
    layers=[
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

def get_fine_to_coarse(fine_to_coarse_path):
  """Map fine label indexing to coarse label indexing."""
  with open(fine_to_coarse_path, "rb") as f:
    d = pickle.load(f)
  fine_to_coarse_dict = d["fine_index_to_coarse_index"]
  fine_to_coarse_dict[255] = -1
  fine_to_coarse_map = np.vectorize(
      lambda x: fine_to_coarse_dict[x]
  )  # not in-place.
  return fine_to_coarse_map

def load_imdb(imdb_path):
  with open(imdb_path, "r") as f:
    imdb = tuple(f)
    imdb = [id_.rstrip() for id_ in imdb]
    return imdb

def create_path(root, file_list, split="val"):
  """This function creates data loading paths."""
  image_path = []
  label_path = []

  image_folder = split + "2017"
  label_folder = "annotation_" + split + "2017"

  for file in file_list:
    image_path.append(os.path.join(root, image_folder, file + ".jpg"))
    label_path.append(os.path.join(root, label_folder, file + ".png"))
  return image_path, label_path

def process_image(image_path, label_path):
  """This function reads and resizes images and labels."""
  image = tf.io.read_file(image_path)
  image = tf.io.decode_jpeg(image, 3)
  label = tf.io.read_file(label_path)
  label = tf.io.decode_png(label, 3)

  s = tf.shape(image)
  w, h = s[0], s[1]
  c = tf.minimum(w, h)
  w_start = (w - c) // 2
  h_start = (h - c) // 2
  image = image[w_start : w_start + c, h_start : h_start + c, :]
  label = label[w_start : w_start + c, h_start : h_start + c]
  image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
  label = tf.image.resize(
      label,
      (RESOLUTION, RESOLUTION),
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )
  return image, label

def apply_augmentation(image_batch, label_batch):
  """This function applies image augmentation to batches."""
  return augmenter(image_batch), label_batch

def prepare_dict(image_batch, label_batch):
  return {
      "images": image_batch,
      "labels": label_batch,
  }

def prepare_dataset(image_paths, label_paths, batch_size=1):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
  dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(
      batch_size
  )
  dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
  dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
  return dataset.prefetch(AUTO)