import os
import re
import glob
import pickle
from constants import *
from utils import to_2d, log_transform
from dataclasses import dataclass, field
from torch.utils.data import Dataset


@dataclass
class DataSource:
    """
    Container class characterizing a data source for the model.

    Attributes
    ----------
    name : str
        name of the variable (exactly as is labeled in the source pandas dataframe)
    lags : list[int]
        a list of lags to include (0 is the target time)
    log_transform : bool
        True if the variable should be log transformed before being passed to the model
    is_cat : bool
        True if the variable is categorical
    cat_levels : list[int]
        the category levels
    """
    name: str
    lags: list[int]
    log_transform: bool = False
    is_cat: bool = False
    cat_levels: list[int] = field(default_factory=list)


class DataCollection(Dataset):
    """
    Dataset class for the model.

    Attributes
    ----------
    datasources : list[DataSource]
        list of DataSource objects describing variables to be compiled into the dataset
    daily_path : str
        path to directory containing pandas dataframes of daily data
    stack_path : str
        Path to directory for saving intermediary stacks during sample generation. Saved as a python dictionary
        (str : np.array), with 'X' containing the predictors (n, height, width), 'y' the predictand (1, height, width),
        'X_min' the sample minimums of predictors (n, 1, 1), 'X_max' the sample maximums of predictors (n, 1, 1).
    sample_path : str
        same as stack path except X is shape (n, window_size, window_size), y is shape (1, window_size, window_size),
        and two additional fields 'i' and 'j' are included in the dictionary to denote the location of the top left
        corner of the window
    num_hist_bins : int
        The number of bins when calculating the distribution of the target variable. Used for calculating weights for
        the loss function. Default 100.
    target_var : str
        The name of the variable to be used as the target. Must match column name in daily data. Default "aod_avg".
    fill_value : int
        The value to replace NaN's for both X and y before being fed into the model. Default 0.
    max_lag : int
        the largest lag contained within datasources
    target_is_log : bool
        True if target should be log transformed
    mins : pandas.Series
        Sample minimums of the data. Calculated after log transforms.
    maxs : pandas.Series
        Sample maximums of the data. Calculated after log transforms.
    hist_bin_width
    hist_bin_edges
    hist_bin_counts
    bin_edges
    bin_counts
    bin_weights
    fns
    """
    def __init__(self,
                 datasources,
                 daily_path,
                 stack_path,
                 sample_path,
                 num_hist_bins=100,
                 target_var="aod_avg",
                 fill_value=0,
                 mins_path=None,
                 maxs_path=None,
                 hist_path=None):
        self.datasources = datasources
        self.target_var = target_var
        self.fill_value = fill_value
        self.stack_path = stack_path
        self.sample_path = sample_path
        self.max_lag = 0
        self.mins = None
        self.maxs = None
        self.target_is_log = False
        self.num_hist_bins = num_hist_bins

        for ds in self.datasources:
            curr_max = max(ds.lags)
            if curr_max > self.max_lag:
                self.max_lag = curr_max
            if ds.name == target_var and ds.log_transform:
                self.target_is_log = True

        if mins_path is not None and maxs_path is not None:
            self.mins = pd.read_pickle(mins_path)
            self.maxs = pd.read_pickle(maxs_path)
        else:
            # Calculate minimums and maximums
            for f in glob.glob(f"{daily_path}*.pkl"):
                print(f)
                df = pd.read_pickle(f)
                for ds in self.datasources:
                    if ds.log_transform:
                        df[ds.name] = log_transform(df[ds.name])
                if self.mins is None:
                    self.mins = df.min()
                    self.maxs = df.max()
                else:
                    df.loc[len(df)] = self.mins
                    df.loc[len(df)] = self.maxs
                    self.mins = df.min()
                    self.maxs = df.max()

        if hist_path is not None:
            with open(hist_path, "rb") as file:
                hist_dict = pickle.load(file)
                self.hist_bin_width = hist_dict['width']
                self.hist_bin_edges = hist_dict['edges']
                self.hist_bin_counts = hist_dict['counts']
        else:
            # Calculate histogram counts
            self.hist_bin_width = (self.maxs[target_var] - self.mins[target_var]) / self.num_hist_bins
            self.hist_bin_edges = self.mins[target_var] + (np.arange(self.num_hist_bins + 1) * self.hist_bin_width)
            self.hist_bin_edges[-1] = np.inf
            self.hist_bin_counts = np.zeros(self.num_hist_bins)
            for f in glob.glob(f"{daily_path}*.pkl"):
                df = pd.read_pickle(f)
                target = df[self.target_var].to_numpy()[np.newaxis, ...]
                if self.target_is_log:
                    target = log_transform(target)
                self.hist_bin_counts = self.hist_bin_counts + np.sum(
                    (target >= self.hist_bin_edges.reshape((self.num_hist_bins + 1, 1))[:-1]) & (
                                target < self.hist_bin_edges.reshape((self.num_hist_bins + 1, 1))[1:]), axis=1)

    def set_bin_edges(self, indexes):
        num_weight_bins = len(indexes) - 1
        self.bin_edges = self.hist_bin_edges[indexes]
        indexes = np.array(indexes).reshape((-1, 1))
        hist_idx = np.tile(np.arange(self.num_hist_bins), (num_weight_bins, 1))
        idx = np.where((hist_idx >= indexes[:-1]) & (hist_idx < indexes[1:]), hist_idx, -1)
        counts_p = np.concatenate([self.hist_bin_counts, np.zeros(1)])
        self.bin_counts = counts_p[idx].sum(axis=1)
        self.bin_weights = np.sum(self.bin_counts) / self.bin_counts

    def __get_weights(self, y):
        edges = self.bin_edges.reshape((-1, 1, 1))
        y_weights = self.bin_weights[np.argmax((y >= edges[:-1]) & (y < edges[1:]), axis=0, keepdims=True)]
        y_weights[np.where(np.isnan(y))] = 0
        return y_weights

    def save_samples(self, backlog, date, window_size=WINDOW_SIZE, stride=WINDOW_SIZE // 2, test=False):
        stack = []
        x_min = np.array([[[]]], dtype=np.float32).reshape((0, 1, 1))
        x_max = np.array([[[]]], dtype=np.float32).reshape((0, 1, 1))
        x_log_idx = np.array([], dtype=bool)
        for ds in self.datasources:
            name = ds.name
            for lag in ds.lags:
                var = to_2d(backlog[lag][name], HEIGHT, WIDTH)

                if ds.is_cat:
                    temp = np.zeros((len(ds.cat_levels), HEIGHT, WIDTH))
                    for i in range(len(ds.cat_levels)):
                        if np.isnan(ds.cat_levels[i]):
                            temp[i] = np.isnan(var).astype(np.float32)
                        else:
                            temp[i] = (var == ds.cat_levels[i]).astype(np.float32)
                    var = temp
                    if name == self.target_var and lag == 0:
                        y = var
                    else:
                        stack.append(var)
                        x_log_idx = np.concatenate([x_log_idx, np.array([False] * len(ds.cat_levels))])
                        x_min = np.concatenate([x_min, np.zeros((len(ds.cat_levels), 1, 1))])
                        x_max = np.concatenate([x_max, np.ones((len(ds.cat_levels), 1, 1))])
                else:
                    if name == self.target_var and lag == 0:
                        y = var
                    else:
                        stack.append(var)
                        if ds.log_transform:
                            x_log_idx = np.concatenate([x_log_idx, np.array([True])])
                        else:
                            x_log_idx = np.concatenate([x_log_idx, np.array([False])])
                        x_min = np.concatenate([x_min, np.array([[[self.mins[name]]]])])
                        x_max = np.concatenate([x_max, np.array([[[self.maxs[name]]]])])

        stack = np.concatenate(stack, axis=0)
        file = open(f"{self.stack_path}{date.strftime('%Y%m%d')}.pkl", "wb")
        pickle.dump({'y': y, 'X': stack, 'X_min': x_min, 'X_max': x_max, 'X_log_idx': x_log_idx}, file)
        file.close()

        stack_windows = \
        np.lib.stride_tricks.sliding_window_view(stack, window_shape=(stack.shape[0], window_size, window_size))[0]
        y_windows = np.lib.stride_tricks.sliding_window_view(y, window_shape=(1, window_size, window_size))[0]

        save_int = 0
        for i in range(0, y_windows.shape[0], stride):
            for j in range(0, y_windows.shape[1], stride):
                if test or np.any(~np.isnan(y_windows[i, j])):
                    with open(f"{self.sample_path}{date.strftime('%Y%m%d')}_{save_int:04}.pkl", "wb") as file:
                        pickle.dump({'y': y_windows[i, j],
                                     'X': stack_windows[i, j],
                                     'X_min': x_min,
                                     'X_max': x_max,
                                     'X_log_idx': x_log_idx,
                                     'i': i,
                                     'j': j}, file)
                        save_int += 1

    def scan_samples(self):
        self.fns = [f for f in os.listdir(self.sample_path) if re.search(r".*.pkl$", f)]

    def __len__(self):
        if not hasattr(self, 'fns'):
            raise NameError("Call scan_samples first.")
        return len(self.fns)

    def __getitem__(self, idx):
        if not hasattr(self, 'fns'):
            raise NameError("Call scan_samples first.")
        date = int(self.fns[idx][:-9])
        with open(f"{self.sample_path}{self.fns[idx]}", "rb") as file:
            d = pickle.load(file)
            x = d['X']
            x_min = d['X_min']
            x_max = d['X_max']
            x_log_idx = d['X_log_idx']
            x[x_log_idx] = log_transform(x[x_log_idx])
            x = (x - x_min) / (x_max - x_min)
            x = np.where(np.isnan(x), self.fill_value, x)
            y = d['y']
            if self.target_is_log:
                y = log_transform(y)
            weights = self.__get_weights(y)
            y_min = self.mins[self.target_var]
            y_max = self.maxs[self.target_var]
            y = (y - y_min) / (y_max - y_min)
            y = np.where(np.isnan(y), self.fill_value, y)
            i = d['i']
            j = d['j']
        return x, y, weights, date, i, j

    def save_mins(self, path):
        self.mins.to_pickle(path)

    def save_maxs(self, path):
        self.maxs.to_pickle(path)

    def save_hist(self, path):
        with open(path, "wb") as file:
            pickle.dump({'width': self.hist_bin_width, 'edges': self.hist_bin_edges, 'counts': self.hist_bin_counts},
                        file)
