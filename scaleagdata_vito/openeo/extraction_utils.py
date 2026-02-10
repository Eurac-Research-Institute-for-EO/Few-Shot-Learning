import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tabulate import tabulate  # type: ignore

BANDS = {
    "S2-L2A": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "S1-SIGMA0": ["VV", "VH"],
    "DEM": ["slope", "elevation"],
    "AGERA5": ["PRECIP", "TMEAN"],
}

PLOT_BAND_MAPPING = {
    "S2-L2A-B02": "B2",
    "S2-L2A-B03": "B3",
    "S2-L2A-B04": "B4",
    "S2-L2A-B05": "B5",
    "S2-L2A-B06": "B6",
    "S2-L2A-B07": "B7",
    "S2-L2A-B08": "B8",
    "S2-L2A-B8A": "B8A",
    "S2-L2A-B11": "B11",
    "S2-L2A-B12": "B12",
    "S1-SIGMA0-VV": "VV",
    "S1-SIGMA0-VH": "VH",
    "slope": "slope",
    "elevation": "elevation",
    "AGERA5-PRECIP": "avg temperature",
    "AGERA5-TMEAN": "precipitation (mm3)",
    "NDVI": "NDVI",
}

NODATAVALUE = 65535


def _apply_band_scaling(array: np.array, bandname: str) -> np.array:
    """Apply scaling to the band values based on the band name.
    Parameters
    ----------
    array : np.array
        array containing the band values
    bandname : str
        name of the band
    Returns
    -------
    np.array
        array containing the scaled band values
    Raises
    ------
    ValueError
        If the band is not supported
    """

    idx_valid = array != NODATAVALUE
    array = array.astype(np.float32)

    # No scaling for DEM bands
    if bandname in BANDS["DEM"]:
        pass
    # Divide by 10000 for S2 bands
    elif bandname.startswith("S2-L2A"):
        array[idx_valid] = array[idx_valid] / 10000
    # Convert to dB for S1 bands
    elif bandname.startswith("S1-SIGMA0"):
        idx_valid = idx_valid & (array > 0)
        array[idx_valid] = 20 * np.log10(array[idx_valid]) - 83
    # Scale meteo bands by factor 100
    elif bandname.startswith("AGERA5"):
        array[idx_valid] = array[idx_valid] / 100
    else:
        raise ValueError(f"Unsupported band name {bandname}")

    return array


def get_band_statistics(
    extractions_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Get the band statistics for the point extractions.
    Parameters
    ----------
    extractions_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the point extractions in LONG format.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the band statistics
    """

    # Get the band statistics
    band_stats = {}
    for sensor, bands in BANDS.items():
        for band in bands:
            if sensor == "DEM":
                bandname = band
            else:
                bandname = f"{sensor}-{band}"
            if bandname in extractions_gdf.columns:
                # count percentage of nodata values
                nodata_count = (extractions_gdf[bandname] == NODATAVALUE).sum()
                nodata_percentage = nodata_count / len(extractions_gdf) * 100
                # Apply scaling
                scaled_values = _apply_band_scaling(
                    extractions_gdf[bandname].values, bandname
                )
                # discard nodata values
                scaled_values = scaled_values[scaled_values != NODATAVALUE]
                # get the statistics and only show 2 decimals
                min = np.min(scaled_values) if len(scaled_values) > 0 else 0
                max = np.max(scaled_values) if len(scaled_values) > 0 else 0
                mean = np.mean(scaled_values) if len(scaled_values) > 0 else 0
                std = np.std(scaled_values) if len(scaled_values) > 0 else 0
                band_stats[bandname] = {
                    "band": bandname,
                    "%_nodata": f"{nodata_percentage:.2f}",
                    "min": f"{min:.2f}",
                    "max": f"{max:.2f}",
                    "mean": f"{mean:.2f}",
                    "std": f"{std:.2f}",
                    "start_datae": extractions_gdf["start_date"].min(),
                    "end_date": extractions_gdf["end_date"].max(),
                }

    # Convert to DataFrame
    stats_df = pd.DataFrame(band_stats).T
    print(tabulate(stats_df, headers="keys", tablefmt="psql", showindex=False))

    return stats_df


def load_point_extractions(
    extractions_dir: Path,
    subset=False,
) -> gpd.GeoDataFrame:
    """Load point extractions from the given folder.

    Parameters
    ----------
    extractions_dir : Path
        path containing extractions for a given collection
    subset : bool, optional
        whether to subset the data to reduce the size, by default False

    Returns
    -------
    GeoPandas GeoDataFrame
        GeoDataFrame containing all point extractions,
        organized in long format
        (each row represents a single timestep for a single sample)
    """

    # Look for all extractions in the given folder
    infiles = list(Path(extractions_dir).glob("**/*.geoparquet"))
    # Get rid of merged geoparquet
    infiles = [f for f in infiles if not Path(f).is_dir()]

    if len(infiles) == 0:
        raise FileNotFoundError(f"No point extractions found in {extractions_dir}")
    logger.info(f"Found {len(infiles)} geoparquet files in {extractions_dir}")

    if subset:
        # only load first file
        gdf = gpd.read_parquet(infiles[0])
    else:
        # load all files
        gdf = gpd.read_parquet(infiles)

    return gdf


def check_job_status(output_folder: Path) -> dict:
    """Check the status of the jobs in the given output folder.

    Parameters
    ----------
    output_folder : Path
        folder where extractions are stored

    Returns
    -------
    dict
        status_histogram
    """

    # Read job tracking csv file
    job_status_df = _read_job_tracking_csv(output_folder)

    # Summarize the status in histogram
    status_histogram = _count_by_status(job_status_df)

    # convert to pandas dataframe
    status_count = pd.DataFrame(status_histogram.items(), columns=["status", "count"])
    status_count = status_count.sort_values(by="count", ascending=False)

    print(tabulate(status_count, headers="keys", tablefmt="psql", showindex=False))

    return status_histogram


def get_succeeded_job_details(output_folder: Path) -> pd.DataFrame:
    """Get details of succeeded extraction jobs in the given output folder.

    Parameters
    ----------
    output_folder : Path
        folder where extractions are stored
    Returns
    -------
    pd.DataFrame
        details of succeeded jobs
    """

    # Read job tracking csv file
    job_status_df = _read_job_tracking_csv(output_folder)

    # Gather metadata on succeeded jobs
    succeeded_jobs = job_status_df[
        job_status_df["status"].isin(["finished", "postprocessing"])
    ].copy()
    if len(succeeded_jobs) > 0:
        # Derive number of features involved in each job
        nfeatures = []
        for i, row in succeeded_jobs.iterrows():
            nfeatures.append(len(json.loads(row["geometry"])["features"]))
        succeeded_jobs.loc[:, "n_samples"] = nfeatures
        # Gather essential columns
        succeeded_jobs = succeeded_jobs[
            [
                "id",
                "s2_tile",
                "n_samples",
                "duration",
            ]
        ]
        # Convert duration to minutes
        # convert NaN to 0 seconds
        succeeded_jobs["duration"] = succeeded_jobs["duration"].fillna("0s")
        seconds = succeeded_jobs["duration"].str.split("s").str[0].astype(int)
        succeeded_jobs["duration"] = seconds / 60
        succeeded_jobs.rename(columns={"duration": "duration_mins"}, inplace=True)
        if succeeded_jobs["duration_mins"].sum() == 0:
            succeeded_jobs.drop(columns=["duration_mins"], inplace=True)
    else:
        succeeded_jobs = pd.DataFrame()
    # Print details of succeeded jobs
    if not succeeded_jobs.empty:
        print(
            tabulate(
                succeeded_jobs,
                headers="keys",
                tablefmt="psql",
                showindex=False,
            )
        )

    return succeeded_jobs


def _read_job_tracking_csv(output_folder: Path) -> pd.DataFrame:
    """Read job tracking csv file.

    Parameters
    ----------
    output_folder : Path
        folder where extractions are stored

    Returns
    -------
    pd.DataFrame
        job tracking dataframe

    Raises
    ------
    FileNotFoundError
        if the job status file is not found in the designated folder
    """
    job_status_file = output_folder / "job_tracking.csv"
    if job_status_file.exists():
        job_status_df = pd.read_csv(job_status_file)
    else:
        raise FileNotFoundError(f"Job status file not found at {job_status_file}")
    return job_status_df


def _count_by_status(job_status_df, statuses: Iterable[str] = ()) -> dict:
    status_histogram = job_status_df.groupby("status").size().to_dict()
    statuses = set(statuses)
    if statuses:
        status_histogram = {k: v for k, v in status_histogram.items() if k in statuses}
    return status_histogram


def compute_ndvi(gdf):
    b4 = _apply_band_scaling(gdf["S2-L2A-B04"].values, "S2-L2A-B04")
    b8 = _apply_band_scaling(gdf["S2-L2A-B08"].values, "S2-L2A-B08")
    ndvi = (b8 - b4) / (b8 + b4)
    return np.nan_to_num(ndvi, nan=0)


def visualize_timeseries(gdf: gpd.GeoDataFrame, sample_id: str) -> None:
    sample_data = (
        gdf[gdf["sample_id"] == sample_id].replace(65535, 0).sort_values(by="timestamp")
    )
    # checked order in code

    months = sample_data["timestamp"].dt.month
    months_mapping = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    fig, axes = plt.subplots(
        nrows=4, ncols=5, figsize=(20, 15)
    )  # Adjust size as needed
    axes = axes.flatten()  # Flatten the 2D array of axes to iterate easily
    x_ticks = [months_mapping[m] for m in months]

    i = 0
    for band, plot_band in PLOT_BAND_MAPPING.items():
        if band != "NDVI":
            data = _apply_band_scaling(sample_data[band].values, band)
        else:
            data = compute_ndvi(sample_data)
        sns.lineplot(
            x=np.arange(len(months)),
            y=data,
            ax=axes[i],
            alpha=0.5,
            linewidth=3,
            color="blue",
        )  # Overlay actual data
        axes[i].set_title(plot_band)
        axes[i].set_xticks(np.arange(len(months)))
        axes[i].set_xticklabels(x_ticks, rotation=90)
        axes[i].grid()
        i += 1
        # axes[i].set_xlim(0, len(sample_norm[..., i]))  # Adjust as needed

    for j in range(i, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Sample ID {sample_id}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
