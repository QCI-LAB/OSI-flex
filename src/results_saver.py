import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


def create_results_dir():
    results_base = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # include timestamp
    results_path = results_base / f"run_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist
    print(f"Results will be saved to: {results_path}")
    return results_path


def save_results(
    phase: np.ndarray,
    shearing_data : np.array,
    costs: np.ndarray,
    shears: np.ndarray,
    config,
    output_dir="results"
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save numeric data
    # phase as .mat and .npy
    savemat(os.path.join(output_dir, "phase.mat"), {"phase": phase})
    np.save(os.path.join(output_dir, "phase.npy"), phase)
    # data_stack ad .mat
    savemat(os.path.join(output_dir, "shearing_data.mat"), {"shearing_data": shearing_data})
    # costs as .mat and .csv
    savemat(os.path.join(output_dir, "costs.mat"), {"costs": costs})
    np.savetxt(os.path.join(output_dir, "costs.csv"), costs, delimiter=",")

    # 3. Save metadata (config + results)
    metadata = {
        "config": config.__dict__,
        "estimated_shears": shears.tolist(),   # convert ndarray → list
        "n_iterations": int(len(costs)),       # make sure it's int
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # 4. Save figures with results
    # Display cost function progression
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(costs)), costs)
    ax.set(xlabel="Iterations", ylabel="Cost", title="Cost function over iterations")
    fig.savefig(os.path.join(output_dir, "cost_plot.png"))

    # Display estimated phase
    fig0, axs0 = plt.subplots(1, 1, figsize=(8, 8))
    im00 = axs0.imshow(phase, cmap="gray")
    axs0.set_title("Reconstructed image")
    fig0.colorbar(im00, ax=axs0, orientation='vertical', fraction=0.046, pad=0.04)
    fig0.savefig(os.path.join(output_dir, "Output phase.png"))

    diff_range = (shearing_data.min(), shearing_data.max())
    diff_img_num = shearing_data.shape[0]
    fig1, axs1 = plt.subplots(1, ncols=diff_img_num, figsize=(12, 8))
    if diff_img_num == 1:
        axs1 = [axs1]

    for i_img in range(diff_img_num):
        im = axs1[i_img].imshow(shearing_data[i_img], cmap="gray", vmin=diff_range[0], vmax=diff_range[1])
        axs1[i_img].set_title(f"Shearing img no. {i_img}")
    fig1.colorbar(im, ax=axs1[:], fraction=0.046, pad=0.04)
    fig1.savefig(os.path.join(output_dir, "shearing phases.png"))

    plt.show()