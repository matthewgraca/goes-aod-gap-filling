import numpy as np
import pandas as pd

DAILY_DIR = "/scratch/general/nfs1/u6049013/daily/"
STACK_DIR = "/scratch/general/nfs1/u6049013/stacks/"
SAMPLE_DIR = "/scratch/general/nfs1/u6049013/samples1/"

MINS_PATH = "/uufs/chpc.utah.edu/common/home/u6049013/mins.pkl"
MAXS_PATH = "/uufs/chpc.utah.edu/common/home/u6049013/maxs.pkl"
HIST_PATH = "/uufs/chpc.utah.edu/common/home/u6049013/hist.pkl"

HEIGHT = 870
WIDTH = 1200
WINDOW_SIZE = 192

EDGES = np.concatenate([[0, 36], np.arange(54, 101, 2)])
FILL_VALUE = 0

CRS = "+proj=lcc +lon_0=-95 +lat_1=12.190"
GRID = pd.read_pickle("/uufs/chpc.utah.edu/common/home/u6049013/grid.pkl")

IN_CHANNELS = 85
ALPHA = 1.0     # loss function weighting parameter, i.e. l = alpha * l_pixel + beta * l_ms-ssim
BETA = 1e-3
LR = 1e-2
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
DATASET_SPLIT = 0.9
SAVE_EVERY = 1
TOTAL_EPOCHS = 1
