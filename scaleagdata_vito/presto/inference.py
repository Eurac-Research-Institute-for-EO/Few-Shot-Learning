from pathlib import Path
from typing import Literal, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from einops import rearrange
from matplotlib.ticker import ScalarFormatter
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import collate_fn
from torch.utils.data import DataLoader

from scaleagdata_vito.presto.datasets_prometheo import (
    InferenceDataset,
    ScaleAgInferenceDataset,
)


class PrestoPredictor:
    def __init__(
        self,
        model: PretrainedPrestoWrapper,
        batch_size: int = 8192,
        task_type: Literal["regression", "binary", "multiclass"] = "regression",
        composite_window: Literal["dekad", "month"] = "dekad",
    ):
        """
        Initialize the PrestoFeatureExtractor with a Presto model.

        Args:
            model (Presto): The Presto model used for feature extraction.
            batch_size (int): Batch size for dataloader.
        """
        self.model = model  # .to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.task_type = task_type
        self.composite_window = composite_window

    def predict(
        self,
        path_to_file: Path,
        upper_bound: Union[float, None] = None,
        lower_bound: Union[float, None] = None,
        mask_path: Union[str, Path, None] = None,
    ) -> np.ndarray:
        cl = ScaleAgInferenceDataset(composite_window=self.composite_window)
        s1_cube, s2_cube, meteo_cube, dem_cube, latlon_cube, timestamps_cube = (
            cl.nc_to_array(path_to_file, mask_path=mask_path)
        )
        ds = InferenceDataset(
            s1_cube, s2_cube, meteo_cube, dem_cube, latlon_cube, timestamps_cube
        )
        dl = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_probs = []
        self.model.eval()
        for batch in dl:
            with torch.no_grad():
                output = self.model(batch)
                # binary classification
                if self.task_type == "binary":
                    probs = torch.sigmoid(output).cpu().numpy()
                # multiclass classification
                elif self.task_type == "multiclass":
                    probs = torch.softmax(output, dim=-1).cpu().numpy()
                elif self.task_type == "regression":
                    probs = output.cpu().numpy()
                    if upper_bound is not None and lower_bound is not None:
                        probs = revert_to_original_units(
                            probs, upper_bound, lower_bound
                        )
                    else:
                        raise ValueError(
                            "upper_bound and lower_bound used during training"
                            "must be provided for converting results to origininal units"
                        )
                else:
                    raise ValueError(
                        "task_type must be either 'binary', 'multiclass' or 'regression'"
                    )
                all_probs.append(probs.flatten())
        all_probs = np.concatenate(all_probs)
        return all_probs

    def get_predictions(
        self, probs: np.ndarray, threshold: float = 0.5
    ) -> xr.DataArray:
        if self.task_type == "binary":
            preds = probs > threshold
        elif self.task_type == "multiclass":
            preds = probs.argmax(axis=-1)
        else:
            raise ValueError("task_type must be either 'binary' or 'multiclass'")
        return preds


def revert_to_original_units(target_norm, upper_bound, lower_bound):
    return target_norm * (upper_bound - lower_bound) + lower_bound


def reshape_result(result: np.ndarray, path_to_input_file: Path):
    input_arr = xr.load_dataset(path_to_input_file)
    x_coords = input_arr.x.values
    y_coords = input_arr.y.values
    if result.shape[0] != len(x_coords) * len(y_coords):
        class_dim = result.shape[0] // (len(x_coords) * len(y_coords))
        reshaped_result = rearrange(
            result, "(c y x) -> y x c", y=len(y_coords), x=len(x_coords), c=class_dim
        )
    else:
        reshaped_result = rearrange(
            result, "(y x) -> y x", y=len(y_coords), x=len(x_coords)
        )
    return reshaped_result


def min_max_normalize(image):
    # normalize image and set NaNs to NODATA value
    image = np.nan_to_num(image, 65535).astype("uint16")
    return (image - image.min()) / (image.max() - image.min())


def plot_results(path_to_input_file, task, prob_map=None, pred_map=None, ts_index=0):
    rgb = xr.load_dataset(path_to_input_file)
    bands = ["S2-L2A-B04", "S2-L2A-B03", "S2-L2A-B02"]
    rgb = np.stack([rgb[band].values for band in bands], axis=-1)
    if task == "binary":
        if prob_map is None or pred_map is None:
            raise ValueError(
                "prob_map and pred_map must be provided for binary classification"
            )
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)

        axs = [fig.add_subplot(gs[i]) for i in range(3)]
        cax = fig.add_subplot(gs[3])  # dedicated colorbar axis

        axs[0].imshow(min_max_normalize(rgb[ts_index]))
        axs[0].set_title("RGB")
        axs[0].axis("off")
        if task == "binary":
            axs[1].imshow(pred_map, cmap="gray")
        else:
            axs[1].imshow(pred_map, cmap="nipy_spectral")
        axs[1].set_title("Prediction Map")
        axs[1].axis("off")

        im = axs[2].imshow(prob_map, cmap="magma", vmin=0, vmax=1)
        axs[2].set_title("Probability Map")
        axs[2].axis("off")

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks(np.arange(0, 1.1, 0.1))
        plt.show()
    elif task == "multiclass":
        if pred_map is None:
            raise ValueError("pred_map must be provided for multiclass classification")
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

        axs = [fig.add_subplot(gs[i]) for i in range(2)]
        axs[0].imshow(min_max_normalize(rgb[ts_index]))
        axs[0].set_title("RGB")
        axs[0].axis("off")

        axs[1].imshow(pred_map, cmap="nipy_spectral")
        axs[1].set_title("Prediction Map")
        axs[1].axis("off")

        plt.show()
    else:
        if prob_map is None:
            raise ValueError("prob_map must be provided for regression")
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)

        axs = [fig.add_subplot(gs[i]) for i in range(2)]
        cax = fig.add_subplot(gs[2])  # separate axis for colorbar

        axs[0].imshow(min_max_normalize(rgb[ts_index]))
        axs[0].set_title("RGB")
        axs[0].axis("off")

        im = axs[1].imshow(prob_map, cmap="magma", vmin=0, vmax=prob_map.max())
        axs[1].set_title("Prediction Map")
        axs[1].axis("off")

        cbar = fig.colorbar(im, cax=cax)
        # Use scientific (power of 10) format
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(
            (-2, 3)
        )  # show scientific notation for large/small values
        cbar.ax.yaxis.set_major_formatter(formatter)
        plt.show()
