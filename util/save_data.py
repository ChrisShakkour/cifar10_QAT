import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import uuid
from datetime import datetime

# ---------------------------------------------------------------------------
# Globals that make the “one file per Python run” rule work
_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
_FILE_HANDLE: Optional[Path] = None
Number = Union[int, float]

# Global variables for dump_three_lists
def save_and_plot_losses(losses, filepath="losses.npy", png_path="loss_plot.png", y_label='Loss', title='Training Loss', enable_plot=False):
    """
    Saves loss values to .npy file and plots the curve.
    """
    # save the list
    np.save(filepath, np.array(losses))

    # plot
    if enable_plot:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)

# ---------------------------------------------------------------------------
# Plotting 4 arrays in a grid and saving
def plot_4_arrays_save(arr1, arr2, arr3, arr4, save_path="grid_plot.png", cmap="viridis", titles=None):
    """
    Plots 4 list-based arrays on a 2x2 grid and saves the plot.
    
    Args:
        arr1, arr2, arr3, arr4: 2D lists or numpy-like nested lists
        save_path: path to save the resulting plot (e.g., 'plot.png')
        cmap: matplotlib colormap
        titles: optional list of 4 strings for subplot titles
    """

    # Convert to numpy arrays
    arrays = [np.array(a) for a in [arr1, arr2, arr3, arr4]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, arr, idx in zip(axes.flatten(), arrays, range(4)):
        # If the array is 1D, use a line plot. imshow expects 2D image-like data
        # and will raise a TypeError for shapes like (N,) or (2,).
        if arr.ndim == 1:
            ax.plot(arr, marker="o")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            if titles:
                ax.set_title(titles[idx])
            ax.grid(True)
        else:
            im = ax.imshow(arr, cmap=cmap)
            if titles:
                ax.set_title(titles[idx])
            ax.axis("off")
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {save_path}")

# ---------------------------------------------------------------------------
# Globals that make the “one file per Python run” rule work
def dump_three_lists(
    a: List[Number],
    b: List[Number],
    c: List[Number],
    name_a: str,
    name_b: str,
    name_c: str,
    path: Union[str, Path],
    base_name: str = "data",
    log_params: Optional[Dict[str, Any]] = None,  # <‑‑ NEW
) -> Path:
    """
    Write three numeric lists to *one* TXT file per Python run.
    Subsequent calls in the same run overwrite the file.

    Parameters
    ----------
    log_params : dict, optional
        If given, each key–value pair is written at the top of the file
        as "key: value" before the column headers.
    """
    global _FILE_HANDLE

    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)

    if _FILE_HANDLE is None:                           # first call this run
        _FILE_HANDLE = path / f"{base_name}_{_RUN_ID}.txt"

    if _FILE_HANDLE.exists():                          # overwrite on repeat
        _FILE_HANDLE.unlink()

    # Normalise list lengths
    max_len = max(len(a), len(b), len(c))
    pad = lambda lst: lst + [""] * (max_len - len(lst))
    a_p, b_p, c_p = map(pad, (a, b, c))

    # ---------------- write file -----------------
    with _FILE_HANDLE.open("w", encoding="utf-8") as f:
        # 1. Optional run‑level parameters
        if log_params:
            for k, v in log_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")                # blank line before the table

        # 2. Column headers
        f.write(f"{name_a}\t{name_b}\t{name_c}\n")

        # 3. Rows
        for x, y, z in zip(a_p, b_p, c_p):
            f.write(f"{x}\t{y}\t{z}\n")

    return _FILE_HANDLE

# ---------------------------------------------------------------------------
# Initialize results file for current run
def init_results_file(
    path: Union[str, Path],
    base_name: str = "data_repeat",
    log_params: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create/truncate one TXT file for this run, write run-level params
    and the three-column header.
    Returns the full path to the file.
    """
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{base_name}_{_RUN_ID}.txt"
    with file_path.open("w", encoding="utf-8") as f:
        # 1) run-level parameters
        if log_params:
            for k, v in log_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        # 2) column headers
        f.write("train_top1\ttest_top1\tC\n")

    return file_path

# ------------------------------------------------------------------
# Append one row of results to the file created by init_results_file.
def append_result(
    file_path: Union[str, Path],
    train_val: float,
    test_val: float,
    c_value: Any
) -> None:
    """
    Append one row of results to the file created by init_results_file.
    `c_value` can be a list, tuple, scalar, etc (it will be str()-ed).
    """
    with Path(file_path).open("a", encoding="utf-8") as f:
        f.write(f"{train_val}\t{test_val}\t{c_value}\n")