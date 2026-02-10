"""Extract S1, S2, METEO and DEM point data using OpenEO-GFMAP package."""

import logging
import os
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, List, Literal, Optional, Union

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
import requests
from openeo.rest import OpenEoApiError, OpenEoApiPlainError, OpenEoRestError
from openeo_gfmap import (
    Backend,
    BackendContext,
    BoundingBoxExtent,
    FetchType,
    TemporalContext,
)
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_s2grid
from tqdm import tqdm

from scaleagdata_vito.openeo.preprocessing import scaleag_preprocessed_inputs

# Logger used for the pipeline
pipeline_log = logging.getLogger("extraction_pipeline")

pipeline_log.setLevel(level=logging.INFO)

stream_handler = logging.StreamHandler()
pipeline_log.addHandler(stream_handler)

formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
stream_handler.setFormatter(formatter)


class ExtractionCollection(Enum):
    """Collections that can be extracted in the extraction scripts."""

    SAMPLE_SCALEAG = "SAMPLE_SCALEAG"


# Exclude the other loggers from other libraries
class ManagerLoggerFilter(logging.Filter):
    """Filter to only accept the OpenEO-GFMAP manager logs."""

    def filter(self, record):
        return record.name in [pipeline_log.name]


stream_handler.addFilter(ManagerLoggerFilter())


def upload_geoparquet_artifactory(
    gdf: gpd.GeoDataFrame, name: str, collection: str = ""
) -> str:
    """Upload the given GeoDataFrame to artifactory and return the URL of the
    uploaded file. Necessary as a workaround for Polygon sampling in OpenEO
    using custom CRS.
    """
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile()
    gdf.to_parquet(temporary_file.name)

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    if not artifactory_username or not artifactory_password:
        raise ValueError(
            "Artifactory credentials not found. Please set ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD."
        )

    headers = {"Content-Type": "application/octet-stream"}

    upload_url = f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap-temp/openeogfmap_dataframe_{collection}{name}.parquet"

    with open(temporary_file.name, "rb") as f:
        response = requests.put(
            upload_url,
            headers=headers,
            data=f,
            auth=(artifactory_username, artifactory_password),
            timeout=180,
        )

    response.raise_for_status()

    return upload_url


def get_job_nb_polygons(row: pd.Series) -> int:
    """Get the number of polygons in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )


def generate_output_path_sample_scaleag(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
):
    """
    For geometry extractions, only one asset (a geoparquet file) is generated per job.
    Therefore geometry_index is always 0.
    It has to be included in the function signature to be compatible with the GFMapJobManager.
    """

    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])

    # Create the subfolder to store the output
    subfolder = root_folder / utm_zone / s2_tile_id
    subfolder.mkdir(parents=True, exist_ok=True)

    return (
        subfolder
        / f"SCALEAG_{root_folder.name}_{row.start_date}_{row.end_date}_{s2_tile_id}_{row.id}{row.out_extension}"
    )


def create_job_dataframe_sample_scaleag(
    backend: Backend,
    split_jobs: List[gpd.GeoDataFrame],
    start_date_user: Union[str, None],
    end_date_user: Union[str, None],
    composite_window: str = "dekad",
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []

    for job in tqdm(split_jobs):
        s2_tile = job.tile.iloc[0]

        job["lat"] = job.geometry.centroid.y
        job["lon"] = job.geometry.centroid.x

        if "date" in job.columns:
            job.rename(columns={"date": "original_date"}, inplace=True)
            job["original_date"] = pd.to_datetime(job["original_date"])

            min_time = job.original_date.min()
            max_time = job.original_date.max()

            # 9 months before and after the valid time
            start_date = min_time - pd.Timedelta(days=275)
            end_date = max_time + pd.Timedelta(days=275)

            # set again as string so that it is json serializable
            job["original_date"] = job.original_date.dt.strftime("%Y-%m-%d")
        else:
            if start_date_user is None or end_date_user is None:
                raise ValueError(
                    "Start and end dates are required when no date column is present."
                )
            else:
                start_date = datetime.strptime(start_date_user, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_user, "%Y-%m-%d")

        # ensure start date is 1st day of month, end date is last day of month
        start_date = start_date.replace(day=1)
        end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
        )

        job["start_date"] = start_date
        job["end_date"] = end_date

        variables = {
            "backend_name": backend.value,
            "out_prefix": "geometry-extraction",
            "out_extension": ".geoparquet",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "geometry": job.to_json(),
            "composite_window": composite_window,
        }

        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_job_sample_scaleag(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "5G",
    python_memory: str = "2G",
    max_executors: int = 22,
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Load the temporal and spatial extent
    temporal_extent = TemporalContext(row.start_date, row.end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Try to get s2 tile ID to filter the collection
    if "s2_tile" in row:
        pipeline_log.debug(f"Extracting data for S2 tile {row.s2_tile}")
        s2_tile = row.s2_tile
    else:
        s2_tile = None

    inputs = scaleag_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        temporal_extent=temporal_extent,
        composite_window=row.composite_window,
        s2_tile=s2_tile,
    )

    # Finally, create a vector cube based on the Point geometries
    cube = inputs.aggregate_spatial(geometries=geometry, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_polygons(row)
    if pipeline_log is not None:
        pipeline_log.debug("Number of polygons to extract %s", number_points)

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": executor_memory,
        "python-memory": python_memory,
        "executor-cores": "1",
        "max-executors": max_executors,
        "soft-errors": "true",
    }

    return cube.create_job(
        out_format="Parquet",
        title=f"ScaleAgData_Geometry_Extraction_{row.s2_tile}",
        job_options=job_options,
    )


def post_job_action_sample_scaleag(
    job_items: List[pystac.Item],
    row: pd.Series,
    parameters: Optional[dict] = None,
) -> list:
    for idx, item in enumerate(job_items):
        item_asset_path = Path(list(item.assets.values())[0].href)

        gdf = gpd.read_parquet(item_asset_path)

        # Convert the dates to datetime format
        gdf["timestamp"] = pd.to_datetime(gdf["date"])
        gdf.drop(columns=["date"], inplace=True)

        # Convert band dtype to uint16 (temporary fix)
        # TODO: remove this step when the issue is fixed on the OpenEO backend
        bands = [
            "S2-L2A-B02",
            "S2-L2A-B03",
            "S2-L2A-B04",
            "S2-L2A-B05",
            "S2-L2A-B06",
            "S2-L2A-B07",
            "S2-L2A-B08",
            "S2-L2A-B8A",
            "S2-L2A-B11",
            "S2-L2A-B12",
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
            "elevation",
            "slope",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        gdf.to_parquet(item_asset_path, index=False)

    return job_items


def load_dataframe(df_path: Path) -> gpd.GeoDataFrame:
    """Load the input dataframe from the given path."""
    pipeline_log.info("Loading input dataframe from %s.", df_path)

    if df_path.name.endswith(".geoparquet"):
        return gpd.read_parquet(df_path)
    else:
        return gpd.read_file(df_path)


def prepare_job_dataframe(
    input_df: gpd.GeoDataFrame,
    collection: ExtractionCollection,
    max_locations: int,
    backend: Backend,
    start_date: str,
    end_date: str,
    composite_window: str,
) -> gpd.GeoDataFrame:
    """Prepare the job dataframe to extract the data from the given input
    dataframe."""
    pipeline_log.info("Preparing the job dataframe.")
    pipeline_log.info("Performing splitting by s2 grid...")
    split_dfs = split_job_s2grid(input_df, max_points=max_locations)

    pipeline_log.info("Dataframes split to jobs, creating the job dataframe...")
    collection_switch: dict[ExtractionCollection, Callable] = {
        ExtractionCollection.SAMPLE_SCALEAG: create_job_dataframe_sample_scaleag,
    }

    create_job_dataframe_fn = collection_switch.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    job_df = create_job_dataframe_fn(
        backend, split_dfs, start_date, end_date, composite_window
    )
    pipeline_log.info("Job dataframe created with %s jobs.", len(job_df))

    return job_df


def setup_extraction_functions(
    collection: ExtractionCollection,
    memory: str,
    python_memory: str,
    max_executors: int,
) -> tuple[Callable, Callable, Callable]:
    """Setup the datacube creation, path generation and post-job action
    functions for the given collection. Returns a tuple of three functions:
    1. The datacube creation function
    2. The output path generation function
    3. The post-job action function
    """

    datacube_creation = {
        ExtractionCollection.SAMPLE_SCALEAG: partial(
            create_job_sample_scaleag,
            executor_memory=memory,
            python_memory=python_memory,
            max_executors=max_executors,
        ),
    }

    datacube_fn = datacube_creation.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    path_fns = {
        ExtractionCollection.SAMPLE_SCALEAG: partial(
            generate_output_path_sample_scaleag
        ),
    }

    path_fn = path_fns.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    post_job_actions = {
        ExtractionCollection.SAMPLE_SCALEAG: partial(
            post_job_action_sample_scaleag,
        ),
    }

    post_job_fn = post_job_actions.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    return datacube_fn, path_fn, post_job_fn


def manager_main_loop(
    manager: GFMAPJobManager,
    collection: ExtractionCollection,
    job_df: gpd.GeoDataFrame,
    datacube_fn: Callable,
    tracking_df_path: Path,
) -> None:
    """Main loop for the job manager, re-running it whenever an uncatched
    OpenEO exception occurs, and notifying the user through the Pushover API
    whenever the extraction start or an error occurs.
    """
    latest_exception_time = None
    exception_counter = 0

    while True:
        pipeline_log.info("Launching the jobs manager.")
        try:
            manager.run_jobs(job_df, datacube_fn, tracking_df_path)
            return
        except (
            OpenEoApiPlainError,
            OpenEoApiError,
            OpenEoRestError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ) as e:
            pipeline_log.exception("An error occurred during the extraction.\n%s", e)
            if latest_exception_time is None:
                latest_exception_time = pd.Timestamp.now()
                exception_counter += 1
            # 30 minutes between each exception
            elif (datetime.now() - latest_exception_time).seconds < 1800:
                exception_counter += 1
            else:
                latest_exception_time = None
                exception_counter = 0

            if exception_counter >= 3:
                pipeline_log.error(
                    "Too many OpenEO exceptions occurred in a short amount of time, stopping the extraction..."
                )
                raise e
        except Exception as e:
            pipeline_log.exception(
                "An unexpected error occurred during the extraction.\n%s", e
            )
            raise e


def extract(args):
    # Fetches values and setups hardocded values
    collection = args.collection
    max_locations_per_job = args.max_locations
    backend = Backend.CDSE

    if not args.output_folder.is_dir():
        args.output_folder.mkdir(parents=True, exist_ok=True)
        # raise ValueError(f"Output folder {args.output_folder} does not exist.")

    tracking_df_path = Path(args.output_folder) / "job_tracking.csv"

    # # Load the input dataframe and build the job dataframe
    # if args.routine == "training":
    #     assert args.input_df != "", "Input dataframe is required for the training routine."
    input_df = load_dataframe(args.input_df)

    # if input_df[args.unique_id_column] != "":
    input_df["sample_id"] = input_df[args.unique_id_column]
    assert input_df["sample_id"].is_unique, "The unique ID column is not unique."

    job_df = None
    if not tracking_df_path.exists():
        job_df = prepare_job_dataframe(
            input_df,
            collection,
            max_locations_per_job,
            backend,
            args.start_date,
            args.end_date,
            args.composite_window,
        )

    # Setup the extraction functions
    pipeline_log.info("Setting up the extraction functions.")
    datacube_fn, path_fn, post_job_fn = setup_extraction_functions(
        collection,
        args.memory,
        args.python_memory,
        args.max_executors,
    )

    # Initialize and setups the job manager
    pipeline_log.info("Initializing the job manager.")

    job_manager = GFMAPJobManager(
        output_dir=args.output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        poll_sleep=60,
        n_threads=4,
        restart_failed=args.restart_failed,
        stac_enabled=False,
    )

    job_manager.add_backend(
        Backend.CDSE.value,
        cdse_connection,
        parallel_jobs=args.parallel_jobs,
    )

    manager_main_loop(job_manager, collection, job_df, datacube_fn, tracking_df_path)

    pipeline_log.info("Extraction completed successfully.")


def generate_input_for_extractions(input_dict):
    start_date = (
        None if "start_date" not in input_dict.keys() else input_dict["start_date"]
    )
    end_date = None if "end_date" not in input_dict.keys() else input_dict["end_date"]

    job_inputs = pd.Series(
        {
            "collection": ExtractionCollection.SAMPLE_SCALEAG,
            "output_folder": Path(input_dict["output_folder"]),
            "input_df": Path(input_dict["input_df"]),
            "start_date": start_date,
            "end_date": end_date,
            "max_locations": 50,
            "memory": "1800m",
            "executor-memory": "3G",
            "python_memory": "3G",
            "max_executors": 22,
            "parallel_jobs": 2,
            "soft-errors": 0.8,
            "restart_failed": True,
            "unique_id_column": input_dict["unique_id_column"],
            "composite_window": input_dict["composite_window"],
        }
    )

    return job_inputs


def generate_extraction_job_command(
    job_params, extraction_script_path="scaleag-vito/scripts/extractions/extract.py"
):
    command = ["python", extraction_script_path]

    for key, value in job_params.items():
        if isinstance(value, bool):
            if value:  # Add flag if True
                command.append(f"--{key}")
        elif value is not None:  # Skip None values
            if key in ["output_folder", "input_df"]:
                command.extend([f"-{key}", str(value)])
            else:
                command.extend([f"--{key}", str(value)])
    command = " ".join(command)
    print(command)


def collect_inputs_for_inference(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_path: Union[Path, str],
    output_filename: str,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
    composite_window: Literal["month", "dekad"] = "dekad",
):
    """Function to retrieve preprocessed inputs that are being
    used in the generation of WorldCereal products.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    output_path : Union[Path, str]
        output path to download the product to
    backend_context : BackendContext
        backend to run the job on, by default CDSE
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128
        so it uses the OpenEO default setting.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None
    """

    # Preparing the input cube for the inference
    inputs = scaleag_preprocessed_inputs(
        connection=cdse_connection(),
        backend_context=backend_context,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
        composite_window=composite_window,
        fetch_type=FetchType.TILE,
    )

    inputs = inputs.filter_bbox(dict(spatial_extent))

    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "1g",
        "executor-memoryOverhead": "1g",
        "python-memory": "2g",
        "soft-errors": "true",
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    outputfile = Path(output_path) / f"{output_filename}"
    inputs.execute_batch(
        outputfile=outputfile,
        out_format="NetCDF",
        title="ScaleAgData collect inference inputs",
        description="Job that collects inputs for ScaleAg inference",
        job_options=job_options,
    )
