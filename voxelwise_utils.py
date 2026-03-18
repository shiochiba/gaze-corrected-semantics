# %%
import os
import numpy as np
from scipy.stats import zscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import check_cv

from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend

from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.utils import zscore_runs, generate_leave_one_run_out
from voxelwise_tutorials.io import load_hdf5_array, get_data_home
from voxelwise_tutorials.viz import plot_flatmap_from_mapper

import matplotlib.pyplot as plt

directory = get_data_home(dataset='shortclips')
print(directory)

# %%
def load_fmri_data(directory, subject):
    file_name = os.path.join(directory, 'responses', f'{subject}_responses.hdf')

    # Load
    Y_train = load_hdf5_array(file_name, key="Y_train")
    Y_test = load_hdf5_array(file_name, key="Y_test")
    run_onsets = load_hdf5_array(file_name, key="run_onsets")

    # Z-score each run independently
    Y_train = zscore_runs(Y_train, run_onsets)

    # Average test repeats and z-score
    Y_test = zscore(Y_test.mean(0), axis=0)

    # Fill NaNs with 0
    Y_train = np.nan_to_num(Y_train)
    Y_test = np.nan_to_num(Y_test)

    return Y_train, Y_test, run_onsets

# %%
# Load feature space from hdf5 and convert to float32

def load_feature_space(feature_file):
    X_train = load_hdf5_array(feature_file, key="X_train").astype("float32")
    X_test = load_hdf5_array(feature_file, key="X_test").astype("float32")
    return X_train, X_test

# %%
# Set backend

def set_default_backend():
    return set_backend("torch_cuda", on_error="warn")

# %%
# Fit a VEM with kernel ridge regression

def fit_kernel_ridge(X_train, X_test, Y_train, Y_test, cv, delays=[1, 2, 3, 4], alphas=np.logspace(1, 20, 20)):
    scaler = StandardScaler(with_mean=True, with_std=False)
    delayer = Delayer(delays=delays)

    kernel_ridge_cv = KernelRidgeCV(
        alphas=alphas,
        cv=cv,
        solver_params=dict(
            n_targets_batch=500,
            n_alphas_batch=5, 
            n_targets_batch_refit=100
        )
    )

    pipeline = make_pipeline(scaler, delayer, kernel_ridge_cv)
    _ = pipeline.fit(X_train, Y_train)

    scores = pipeline.score(X_test, Y_test)
    return pipeline, scores

# %%
# visualize a model's scores

def plot_single_model_flatmap(scores, subject, directory, vmin=None, vmax=None):
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = scores.max()
        
    mapper_file = os.path.join(directory, "mappers", f"{subject}_mappers.hdf")
    ax = plot_flatmap_from_mapper(scores, mapper_file, vmin=vmin, vmax=vmax)
    plt.show()



