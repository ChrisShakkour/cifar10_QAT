import numpy as np
import matplotlib.pyplot as plt

def plot_four_arrays(arr1, arr2, arr3, arr4, 
                     labels=None, 
                     title="Plot of Four Arrays",
                     xlabel="Index",
                     ylabel="Value"):

    arrays = [arr1, arr2, arr3, arr4]

    if labels is None:
        labels = [f"Array {i+1}" for i in range(4)]

    plt.figure()
    for arr, label in zip(arrays, labels):
        plt.plot(arr, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_five_arrays(arr1, arr2, arr3, arr4, arr5,
                     labels=None, 
                     title="Plot of five Arrays",
                     xlabel="Top1 Accuracy",
                     ylabel="Steps on a"):

    arrays = [arr1, arr2, arr3, arr4, arr5]

    if labels is None:
        labels = [f"Array {i+1}" for i in range(5)]

    plt.figure()
    for arr, label in zip(arrays, labels):
        plt.plot(arr, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./{}.png".format(title), dpi=600)
    #plt.show()


def plot_from_npy_files(baseline_array, file1, file2, file3, file4,
                        labels=None,
                        title="Plot of Four Arrays from .npy Files",
                        xlabel="Index",
                        ylabel="Value"):
    """
    Load four .npy files and plot them on the same figure.
    """
    arr0 = baseline_array
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr3 = np.load(file3)
    arr4 = np.load(file4)

    plot_five_arrays(
        arr0, arr1, arr2, arr3, arr4,
        labels=labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )


def main_cifar10():
    plot_from_npy_files(
        baseline_array = np.full(200, 99.77),
        file1="/home/chris/git/cifar10_QAT/out/MyProject_20251217-010600_baseline_training_200/train_top1.npy",
        file2="/home/chris/git/cifar10_QAT/out/MyProject_20251216-225657_dorefa_with_ste_200/train_top1.npy",
        file3="/home/chris/git/cifar10_QAT/out/MyProject_20251217-222443_dorefa_with_pwl_200/train_top1.npy",
        file4="/home/chris/git/cifar10_QAT/out/MyProject_20251218-050134_dorefa_with_dwl_200/train_top1.npy",
        labels=["Baseline 32-bit validation",
                "Baseline 32-bit training",
                "DoReFa 1-bit with Regular STE",
                "DoReFa 1-bit with PWL STE",
                "DoReFa 1-bit with DWL STE"],
        title="train_top1_compare"
    )
    plot_from_npy_files(
        baseline_array = np.full(200,91.73),
        file1="/home/chris/git/cifar10_QAT/out/MyProject_20251217-010600_baseline_training_200/test_top1.npy",
        file2="/home/chris/git/cifar10_QAT/out/MyProject_20251216-225657_dorefa_with_ste_200/test_top1.npy",
        file3="/home/chris/git/cifar10_QAT/out/MyProject_20251217-222443_dorefa_with_pwl_200/test_top1.npy",
        file4="/home/chris/git/cifar10_QAT/out/MyProject_20251218-050134_dorefa_with_dwl_200/test_top1.npy",
        labels=["Baseline 32-bit validation",
                "Baseline 32-bit training",
                "DoReFa 1-bit with Regular STE",
                "DoReFa 1-bit with PWL STE",
                "DoReFa 1-bit with DWL STE"],
        title="test_top1_compare"
    )


if __name__ == "__main__":
    main_cifar10()
    #main_imagenet()