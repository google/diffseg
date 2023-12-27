# -*- coding: utf-8 -*-
"""cityscapes_data.py
Data pre-processing and loading pipeline for Cityscapes.
"""

import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import glob

RESOLUTION = 512
AUTO = tf.data.AUTOTUNE

augmenter = keras.Sequential(
    layers=[
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

def get_fine_to_coarse():
  """Map fine label indexing to coarse label indexing."""
  label_dict = {
      0: -1,
      1: -1,
      2: -1,
      3: -1,
      4: -1,
      5: -1,
      6: -1,
      7: 0,
      8: 1,
      9: 2,
      10: 3,
      11: 4,
      12: 5,
      13: 6,
      14: 7,
      15: 8,
      16: 9,
      17: 10,
      18: 11,
      19: 12,
      20: 13,
      21: 14,
      22: 15,
      23: 16,
      24: 17,
      25: 18,
      26: 19,
      27: 20,
      28: 21,
      29: 22,
      30: 23,
      31: 24,
      32: 25,
      33: 26,
      -1: -1,
  }
  cityscape_labelmap = np.vectorize(lambda x: label_dict[x])
  return cityscape_labelmap

def create_path(root, split="val"):
  """This function creates data loading paths."""
  image_path = []
  label_path = []

  image_folder = "leftImg8bit/" + split
  label_folder = "gtFine/" + split

  for folder in os.listdir(os.path.join(root, image_folder)):
    for file_path in glob.glob(
        os.path.join(root, image_folder, folder, "*.png")
    ):
      image_path.append(file_path)
      label_path.append(file_path.replace("leftImg8bit","gtFine").replace(".png","_labelIds.png"))
  return image_path, label_path

def process_image(image_path, label_path):
  """This function reads and resizes images and labels."""
  image = tf.io.read_file(image_path)
  image = tf.io.decode_png(image, 3)
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

def prepare_dataset(image_paths, label_paths, batch_size=1, train=False):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
  if train:
    dataset = dataset.shuffle(batch_size * 10)
  dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(
      batch_size
  )
  dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
  dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
  return dataset.prefetch(AUTO)