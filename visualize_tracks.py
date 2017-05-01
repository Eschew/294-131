import numpy as np
import matplotlib.pyplot as plt

from config import *
from util import load_video

def visualize_track(video, track):
  rows, cols = 2, 4
  num_frames = rows*cols
  for i in range(int(np.ceil(len(video)/num_frames))):
    fig, axarr = plt.subplots(rows, cols, figsize=(16, 12))
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                wspace=0, hspace=0)
    for j in range(num_frames):
        if i*num_frames + j >= len(video):
          continue
        im = video[i*num_frames + j]
        bbox = track[i*num_frames + j]
        x1, y1, x2, y2 = bbox

        ax = axarr[j/cols][j%cols]
        ax.set_axis_off()
        ax.imshow(im, aspect='equal')
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
            edgecolor='red', linewidth=3.5))
        ax.text(x1, y1 - 2, '{:s} {:.3f}'.format('detection', score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=8, color='white')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':
  videos = [load_video("ZFSspVdQ_1M=0")]
  tracks = np.load(os.path.join(OUTPUT_ROOT, 'tracks.npy'))

  plt.ion()
  for i in range(len(videos)):
    visualize_track(videos[i], tracks[i])
  plt.ioff()
  plt.show()
