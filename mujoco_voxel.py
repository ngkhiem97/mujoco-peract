import numpy as np
import os
import sys
import shutil
import pickle

import matplotlib
import matplotlib.pyplot as plt

from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs

data_path = "/home/khiem/Dropbox/Projects/Personal/mujoco-voxel/data/colab_dataset/open_drawer/all_variations/episodes"
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
IMAGE_SIZE =  128

# what to visualize
episode_idx_to_visualize = 1 # out of 10 demos
ts = 50 # timestep out of total timesteps

# get demo
demo = get_stored_demo(data_path=data_path,
                      index=episode_idx_to_visualize)     

# print the demo
print(demo)

# extract obs at timestep                 
obs_dict = extract_obs(demo._observations[ts], CAMERAS, t=ts)

# print the obs
print(obs_dict)

# total timesteps in demo
print(f"Demo {episode_idx_to_visualize} | {len(demo._observations)} total steps\n")

# plot rgb and depth at timestep
fig = plt.figure(figsize=(20, 10))
rows, cols = 2, len(CAMERAS)

plot_idx = 1
for camera in CAMERAS:
  # rgb
  rgb_name = "%s_%s" % (camera, 'rgb')
  rgb = np.transpose(obs_dict[rgb_name], (1, 2, 0))
  fig.add_subplot(rows, cols, plot_idx)
  plt.imshow(rgb)
  plt.axis('off')
  plt.title("%s_rgb | step %s" % (camera, ts))

  # depth
  depth_name = "%s_%s" % (camera, 'depth')
  depth = np.transpose(obs_dict[depth_name], (1, 2, 0)).reshape(IMAGE_SIZE, IMAGE_SIZE)
  fig.add_subplot(rows, cols, plot_idx+len(CAMERAS))
  plt.imshow(depth)
  plt.axis('off')
  plt.title("%s_depth | step %s" % (camera, ts))

  plot_idx += 1

plt.show()