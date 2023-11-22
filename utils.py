#Copyright 2023 Google LLC

#Use of this source code is governed by an MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
import numpy as np

def process_image(image_path):
    with open(image_path, "rb") as f:
      image = np.array(PIL.Image.open(f))
      s = tf.shape(image)
      w, h = s[0], s[1]   
      c = tf.minimum(w, h)
      w_start = (w - c) // 2
      h_start = (h - c) // 2
      image = image[w_start : w_start + c, h_start : h_start + c, :]
      image = tf.image.resize(image, (512, 512))
      return image

augmenter = keras.Sequential(
    layers=[
        tf.keras.layers.CenterCrop(512, 512),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

def find_edges(M):
  edges = np.zeros((512,512))
  m1 = M[1:510,1:510] != M[0:509,1:510]
  m2 = M[1:510,1:510] != M[2:511,1:510]
  m3 = M[1:510,1:510] != M[1:510,0:509]
  m4 = M[1:510,1:510] != M[1:510,2:511]
  edges[1:510,1:510] = (m1 | m2 | m3 | m4).astype(int)
  x_new = np.linspace(0, 511, 512)
  y_new = np.linspace(0, 511, 512)
  x_new,y_new=np.meshgrid(x_new,y_new)
  x_new = x_new[edges==1]
  y_new = y_new[edges==1]
  return x_new,y_new

def vis_without_label(M,image,index=None,save=False,dir=None,num_class=26):
  fig = plt.figure(figsize=(20, 20))
  ax = plt.subplot(1, 3, 1)
  ax.imshow(image)
  ax.set_title("Input",fontdict={"fontsize":30})
  plt.axis("off")

  x,y = find_edges(M)
  ax = plt.subplot(1, 3, 2)
  ax.imshow(image)
  ax.imshow(M,cmap='jet',alpha=0.5, vmin=-1, vmax=num_class)
  ax.scatter(x,y,color="blue", s=0.5)
  ax.set_title("Overlay",fontdict={"fontsize":30})
  plt.axis("off")

  ax = plt.subplot(1, 3, 3)
  ax.imshow(M, cmap='jet',alpha=0.5, vmin=-1, vmax=num_class),
  ax.set_title("Segmentation",fontdict={"fontsize":30})
  plt.axis("off")

  if save:
    fig.savefig(open(dir+"/example_{}.png".format(index), 'wb'), format='png',bbox_inches='tight', dpi=200)
    plt.close(fig)


def semantic_mask(image, pred, label_to_mask):
  num_fig = len(label_to_mask)
  plt.figure(figsize=(20, 20))
  for i,label in enumerate(label_to_mask.keys()):
      ax = plt.subplot(1, num_fig, i+1)
      image = image.reshape(512*512,-1)
      bin_mask = np.zeros_like(image)
      for mask in label_to_mask[label]:
        bin_mask[(pred.reshape(512*512)==mask).flatten(),:] = 1
      ax.imshow((image*bin_mask).reshape(512,512,-1))
      ax.set_title(label,fontdict={"fontsize":30})
      ax.axis("off")

