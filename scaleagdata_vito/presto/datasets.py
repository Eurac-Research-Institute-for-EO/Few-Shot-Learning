from datetime import datetime, timedelta
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from presto.dataops import (
    BANDS,
    BANDS_GROUPS_IDX,
    NDVI_INDEX,
    NORMED_BANDS,
    NUM_TIMESTEPS,
    S1_S2_ERA5_SRTM,
    S2_RGB_INDEX,
    DynamicWorld2020_2021,
    S2_NIR_10m_INDEX,
)
from presto.masking import BAND_EXPANSION
from torch.utils.data import Dataset

IDX_TO_BAND_GROUPS = {}
for band_group_idx, (key, val) in enumerate(BANDS_GROUPS_IDX.items()):
    for idx in val:
        IDX_TO_BAND_GROUPS[NORMED_BANDS[idx]] = band_group_idx


class ScaleAgBase(Dataset):
    _NODATAVALUE = 65535
    NUM_TIMESTEPS = 12
    BAND_MAPPING = {
        "OPTICAL-B02-ts{}-10m": "B2",
        "OPTICAL-B03-ts{}-10m": "B3",
        "OPTICAL-B04-ts{}-10m": "B4",
        "OPTICAL-B05-ts{}-20m": "B5",
        "OPTICAL-B06-ts{}-20m": "B6",
        "OPTICAL-B07-ts{}-20m": "B7",
        "OPTICAL-B08-ts{}-10m": "B8",
        "OPTICAL-B8A-ts{}-20m": "B8A",
        "OPTICAL-B11-ts{}-20m": "B11",
        "OPTICAL-B12-ts{}-20m": "B12",
        "SAR-VH-ts{}-20m": "VH",
        "SAR-VV-ts{}-20m": "VV",
        "METEO-precipitation_flux-ts{}-100m": "total_precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature_2m",
    }
    STATIC_BAND_MAPPING = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_name: str,
        task: Literal["regression", "binary", "multiclass"],
        num_timesteps: int = NUM_TIMESTEPS,  #
    ):
        self.df = dataframe.replace({np.nan: self._NODATAVALUE})
        self.target_name = target_name
        self.task = task
        self.num_timesteps = num_timesteps

        if self.task == "multiclass":
            self.class_to_index = {
                label: idx for idx, label in enumerate(dataframe[target_name].unique())
            }
            self.index_to_class = {
                idx: label for idx, label in enumerate(dataframe[target_name].unique())
            }

    def __len__(self):
        return self.df.shape[0]

    def get_target(self, row_d: pd.Series) -> int:
        return int(row_d[self.target_name])

    def row_to_arrays(
        self,
        row: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)

        # make sure that month for encoding gets shifted according to
        # the selected timestep positions. Also ensure circular indexing
        month = datetime.strptime(row_d["start_date"], "%Y-%m-%d").month - 1

        eo_data = np.zeros((self.num_timesteps, len(BANDS)))
        # an assumption we make here is that all timesteps for a token
        # have the same masking
        mask = np.zeros((self.num_timesteps, len(BANDS_GROUPS_IDX)))
        for df_val, presto_val in self.BAND_MAPPING.items():
            # retrieve ts for each band column df_val
            values = np.array(
                [float(row_d[df_val.format(t)]) for t in range(self.num_timesteps)]
            )
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(values, nan=self._NODATAVALUE)
            idx_valid = values != self._NODATAVALUE
            if presto_val in ["VV", "VH"]:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling. conversion to celsius is done in the normalization
                values[idx_valid] = values[idx_valid] / 100
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid
            # add values to eo at specifix index. the order followed is the one suggested by BANDS
            eo_data[:, BANDS.index(presto_val)] = values
        for df_val, presto_val in self.STATIC_BAND_MAPPING.items():
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(row_d[df_val], nan=self._NODATAVALUE)
            idx_valid = values != self._NODATAVALUE
            eo_data[:, BANDS.index(presto_val)] = values
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        # check if the visual bands mask is True
        # or nir mask, and adjust the NDVI mask accordingly
        mask[:, NDVI_INDEX] = np.logical_or(
            mask[:, S2_RGB_INDEX], mask[:, S2_NIR_10m_INDEX]
        )

        return (
            self.check(eo_data),
            mask.astype(bool),
            latlon,
            month,
            self.get_target(row_d),
        )

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, mask_per_token, latlon, month, target = self.row_to_arrays(row)

        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        return (
            self.normalize_and_mask(eo),
            np.ones(self.num_timesteps) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            mask_per_variable,
            target,
        )

    @classmethod
    def normalize_and_mask(cls, eo: np.ndarray):
        # TODO: this can be removed
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        normed_eo = S1_S2_ERA5_SRTM.normalize(eo)  # this adds NDVI and normalizes
        # TODO: fix this. For now, we replicate the previous behaviour
        # only keeps the bands present in the data after normalization and sets to 0 the no_data locations
        normed_eo = np.where(eo[:, keep_indices] != cls._NODATAVALUE, normed_eo, 0)
        return normed_eo

    @staticmethod
    def check(array: np.ndarray) -> np.ndarray:
        assert not np.isnan(array).any()
        return array


class ScaleAGDataset(ScaleAgBase):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_name: str,
        task: Literal["regression", "binary", "multiclass"],
        num_timesteps: int = NUM_TIMESTEPS,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ):
        super().__init__(dataframe, target_name, task, num_timesteps)

        # bound label values to valid range
        if task == "regression":
            if upper_bound is None or lower_bound is None:
                upper_bound = dataframe[target_name].max()
                lower_bound = dataframe[target_name].min()
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            dataframe[target_name] = dataframe[target_name].clip(
                lower=lower_bound, upper=upper_bound
            )
            self.mean = np.mean(dataframe[target_name])
            self.std = np.std(dataframe[target_name])

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, mask_per_token, latlon, month, target = self.row_to_arrays(row)
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        if self.task == "regression":
            target = self.normalize_target(target)
        # convert classes to indices for multiclass
        elif self.task == "multiclass":
            target = self.class_to_index[target]

        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.num_timesteps) * (DynamicWorld2020_2021.class_amount),
            latlon,
            self.get_month_array(row) if self.num_timesteps == 36 else month,
            mask_per_variable,
        )

    def get_month_array(self, row: pd.Series) -> np.ndarray:
        start_date, end_date = datetime.strptime(
            row.start_date, "%Y-%m-%d"
        ), datetime.strptime(row.end_date, "%Y-%m-%d")

        # Calculate the step size for 10-day intervals and create a list of dates
        step = int((end_date - start_date).days / (self.num_timesteps - 1))
        date_vector = [
            start_date + timedelta(days=i * step) for i in range(self.num_timesteps)
        ]

        # Ensure last date is not beyond the end date
        if date_vector[-1] > end_date:
            date_vector[-1] = end_date

        return np.array([d.month - 1 for d in date_vector])

    def normalize_target(self, target):
        return (target - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def z_scaling(self, x):
        return (x - self.mean) / self.std

    def revert_to_original_units(self, target_norm):
        return target_norm * (self.upper_bound - self.lower_bound) + self.lower_bound
