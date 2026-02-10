from pathlib import Path
from typing import Any, Dict, List, Optional

from openeo import UDF, Connection, DataCube
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.fetching.generic import build_generic_extractor
from openeo_gfmap.fetching.s1 import build_sentinel1_grd_extractor
from openeo_gfmap.fetching.s2 import build_sentinel2_l2a_extractor
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import compress_backscatter_uint16


def raw_datacube_S2(
    connection: Connection,
    backend_context: BackendContext,
    temporal_extent: TemporalContext,
    bands: List[str],
    fetch_type: FetchType,
    # spatial_extent: Optional[SpatialContext] = None,
    filter_tile: Optional[str] = None,
    distance_to_cloud_flag: Optional[bool] = True,
    additional_masks_flag: Optional[bool] = True,
    apply_mask_flag: Optional[bool] = False,
    tile_size: Optional[int] = None,
) -> DataCube:
    """Extract Sentinel-2 datacube from OpenEO using GFMAP routines.
    Raw data is extracted with no cloud masking applied by default (can be
    enabled by setting `apply_mask=True`). In additional to the raw band values
    a cloud-mask computed from the dilation of the SCL layer, as well as a
    rank mask from the BAP compositing are added.

    Parameters
    ----------
    connection : Connection
        OpenEO connection instance.
    backend_context : BackendContext
        GFMAP Backend context to use for extraction.
    spatial_extent : SpatialContext
        Spatial context to extract data from, can be a GFMAP BoundingBoxExtent,
        a GeoJSON dict or an URL to a publicly accessible GeoParquet file.
    temporal_extent : TemporalContext
        Temporal context to extract data from.
    bands : List[str]
        List of Sentinel-2 bands to extract.
    fetch_type : FetchType
        GFMAP Fetch type to use for extraction.
    filter_tile : Optional[str], optional
        Filter by tile ID, by default disabled. This forces the process to only
        one tile ID from the Sentinel-2 collection.
    apply_mask : bool, optional
        Apply cloud masking, by default False. Can be enabled for high
        optimization of memory usage.
    """
    # Extract the SCL collection only
    scl_cube_properties = {"eo:cloud_cover": lambda val: val <= 95.0}
    if filter_tile:
        scl_cube_properties["tileId"] = lambda val: val == filter_tile

    # Create the job to extract S2
    extraction_parameters: dict[str, Any] = {
        "target_resolution": None,  # Disable target resolution
        "load_collection": {
            "eo:cloud_cover": lambda val: val <= 95.0,
        },
    }

    scl_cube = connection.load_collection(
        collection_id="SENTINEL2_L2A",
        bands=["SCL"],
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        properties=scl_cube_properties,
    )

    # Resample to 10m resolution for the SCL layer
    scl_cube = scl_cube.resample_spatial(10)

    # Compute the SCL dilation mask
    scl_dilated_mask = scl_cube.process(
        "to_scl_dilation_mask",
        data=scl_cube,
        scl_band_name="SCL",
        kernel1_size=17,  # 17px dilation on a 10m layer
        kernel2_size=77,  # 77px dilation on a 10m layer
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3,
    ).rename_labels("bands", ["S2-L2A-SCL_DILATED_MASK"])

    additional_masks = scl_dilated_mask

    if distance_to_cloud_flag:
        # Compute the distance to cloud and add it to the cube
        distance_to_cloud = scl_cube.apply_neighborhood(
            process=UDF.from_file(Path(__file__).parent / "udf_distance_to_cloud.py"),
            size=[
                {"dimension": "x", "unit": "px", "value": 256},
                {"dimension": "y", "unit": "px", "value": 256},
                {"dimension": "t", "unit": "null", "value": "P1D"},
            ],
            overlap=[
                {"dimension": "x", "unit": "px", "value": 16},
                {"dimension": "y", "unit": "px", "value": 16},
            ],
        ).rename_labels("bands", ["S2-L2A-DISTANCE-TO-CLOUD"])

        additional_masks = scl_dilated_mask.merge_cubes(distance_to_cloud)

    if additional_masks_flag:
        extraction_parameters["pre_merge"] = additional_masks

    if filter_tile:
        extraction_parameters["load_collection"]["tileId"] = (
            lambda val: val == filter_tile
        )
    if apply_mask_flag:
        extraction_parameters["pre_mask"] = scl_dilated_mask

    if tile_size is not None:
        extraction_parameters["update_arguments"] = {
            "featureflags": {"tilesize": tile_size}
        }

    extractor = build_sentinel2_l2a_extractor(
        backend_context,
        bands=bands,
        fetch_type=fetch_type,
        **extraction_parameters,
    )

    return extractor.get_cube(connection, None, temporal_extent)


def raw_datacube_S1(
    connection: Connection,
    backend_context: BackendContext,
    temporal_extent: TemporalContext,
    bands: List[str],
    fetch_type: FetchType,
    target_resolution: float = 20.0,
    # spatial_extent: Optional[SpatialContext] = None,
    orbit_direction: Optional[str] = None,
    tile_size: Optional[int] = None,
) -> DataCube:
    """Extract Sentinel-1 datacube from OpenEO using GFMAP routines.

    Parameters
    ----------
    connection : Connection
        OpenEO connection instance.
    backend_context : BackendContext
        GFMAP Backend context to use for extraction.
    spatial_extent : SpatialContext
        Spatial context to extract data from, can be a GFMAP BoundingBoxExtent,
        a GeoJSON dict or an URL to a publicly accessible GeoParquet file.
    temporal_extent : TemporalContext
        Temporal context to extract data from.
    bands : List[str]
        List of Sentinel-1 bands to extract.
    fetch_type : FetchType
        GFMAP Fetch type to use for extraction.
    target_resolution : float, optional
        Target resolution to resample the data to, by default 20.0.
    orbit_direction : Optional[str], optional
        Orbit direction to filter the data, by default None. If None and the
        backend is in CDSE, then querries the catalogue for the best orbit
        direction to use. In the case querrying is unavailable or fails, then
        uses "ASCENDING" as a last resort.
    """
    extractor_parameters: Dict[str, Any] = {
        "target_resolution": target_resolution,
    }

    if orbit_direction is not None:
        extractor_parameters["load_collection"] = {
            "sat:orbit_state": lambda orbit: orbit == orbit_direction,
            "polarisation": lambda pol: pol == "VV&VH",
        }
    else:
        extractor_parameters["load_collection"] = {
            "polarisation": lambda pol: pol == "VV&VH",
        }

    if tile_size is not None:
        extractor_parameters["update_arguments"] = {
            "featureflags": {"tilesize": tile_size}
        }

    extractor = build_sentinel1_grd_extractor(
        backend_context, bands=bands, fetch_type=fetch_type, **extractor_parameters
    )

    return extractor.get_cube(connection, None, temporal_extent)


def raw_datacube_DEM(
    connection: Connection,
    backend_context: BackendContext,
    fetch_type: FetchType,
    # spatial_extent: Optional[SpatialContext] = None,
) -> DataCube:
    extractor = build_generic_extractor(
        backend_context=backend_context,
        bands=["COP-DEM"],
        fetch_type=fetch_type,
        collection_name="COPERNICUS_30",
    )
    """Method to get the DEM datacube from the backend.
    If running on CDSE backend, the slope is also loaded from the global
    slope collection and merged with the DEM cube.

    Returns
    -------
    DataCube
        openEO datacube with the DEM data (and slope if available).
    """

    cube = extractor.get_cube(connection, None, None)
    cube = cube.rename_labels(dimension="bands", target=["elevation"])

    if backend_context.backend == Backend.CDSE:
        # On CDSE we can load the slope from a global slope collection
        # but this currently only works for tile fetching.

        slope = connection.load_stac(
            "https://stac.openeo.vito.be/collections/COPERNICUS30_DEM_SLOPE",
            bands=["Slope"],
        ).rename_labels(dimension="bands", target=["slope"])
        # Client fix for CDSE, the openeo client might be unsynchronized with
        # the backend.
        if "t" not in slope.metadata.dimension_names():
            slope.metadata = slope.metadata.add_dimension("t", "2020-01-01", "temporal")
        slope = slope.min_time()

        # Note that when slope is available we use it as the base cube
        # to merge DEM with, as it comes at 20m resolution.
        cube = slope.merge_cubes(cube)

    return cube


def precomposited_datacube_METEO(
    connection: Connection,
    temporal_extent: TemporalContext,
    # spatial_extent: Optional[SpatialContext] = None,
    period: str = "dekad",
) -> DataCube:
    """Extract the precipitation and temperature AGERA5 data from a
    pre-composited and pre-processed collection. The data is stored in the
    CloudFerro S3 stoage, allowing faster access and processing from the CDSE
    backend.

    Limitations:
        - Only two bands are available: precipitation-flux and temperature-mean.
        - This function does not support fetching points or polygons, but only
          tiles.
    """
    temporal_extent = [temporal_extent.start_date, temporal_extent.end_date]

    if period == "dekad":
        # Dekadal composited METEO data
        cube = connection.load_stac(
            "https://stac.openeo.vito.be/collections/agera5_dekad",  ####
            temporal_extent=temporal_extent,
            bands=["precipitation-flux", "temperature-mean"],
        )
    elif period == "month":
        # Monthly composited METEO data
        cube = connection.load_stac(
            "https://stac.openeo.vito.be/collections/agera5_monthly",
            temporal_extent=temporal_extent,
            bands=["precipitation-flux", "temperature-mean"],
        )
    else:
        raise NotImplementedError(
            f"Period {period} is not supported for precomposited meteo."
        )
    # cube.result_node().update_arguments(featureflags={"tilesize": 1})
    cube = cube.rename_labels(
        dimension="bands", target=["AGERA5-PRECIP", "AGERA5-TMEAN"]
    )

    return cube


def scaleag_preprocessed_inputs(
    connection: Connection,
    backend_context: BackendContext,
    temporal_extent: TemporalContext,
    composite_window: str = "dekad",
    # spatial_extent: Union[GeoJSON, BoundingBoxExtent, str, None] = None,
    fetch_type: Optional[FetchType] = FetchType.POINT,
    disable_meteo: bool = False,
    s1_orbit_state: Optional[str] = None,
    tile_size: Optional[int] = None,
    s2_tile: Optional[str] = None,
) -> DataCube:
    """
    Preprocesses data during OpenEO extraction to prepare inputs for Presto.
    Args:
        connection (Connection): The connection to the backend.
        backend_context (BackendContext): The backend context.
        spatial_extent (SpatialContext): The spatial extent.
        temporal_extent (TemporalContext): The temporal extent.
        period (str, optional): The period for compositing. Defaults to "dekad".
        fetch_type (Optional[FetchType], optional): The fetch type. Defaults to FetchType.POINT.
        disable_meteo (bool, optional): Flag to disable meteorological data. Defaults to False.
        s1_orbit_state (Optional[str], optional): The orbit state for Sentinel-1. Defaults to None.
        tile_size (Optional[int], optional): The tile size. Defaults to None.
    Returns:
        DataCube: The preprocessed data cube.
    """
    # Extraction of S2 from GFMAP
    s2_data = raw_datacube_S2(
        connection=connection,
        backend_context=backend_context,
        temporal_extent=temporal_extent,
        bands=[
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
        ],
        fetch_type=fetch_type,
        filter_tile=s2_tile,
        distance_to_cloud_flag=False,
        additional_masks_flag=False,
        apply_mask_flag=True,
        tile_size=tile_size,
    )

    s2_data = median_compositing(s2_data, period=composite_window)

    # Cast to uint16
    s2_data = s2_data.linear_scale_range(0, 65534, 0, 65534)

    # Extraction of the S1 data
    # Decides on the orbit direction from the maximum overlapping area of
    # available products.
    s1_data = raw_datacube_S1(
        connection=connection,
        backend_context=backend_context,
        temporal_extent=temporal_extent,
        bands=[
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
        ],
        fetch_type=fetch_type,
        target_resolution=20.0,  # Compute the backscatter at 20m resolution, then upsample nearest neighbor when merging cubes
        orbit_direction=s1_orbit_state,  # If None, make the query on the catalogue for the best orbit
        tile_size=tile_size,
    )

    s1_data = mean_compositing(s1_data, period=composite_window)
    s1_data = compress_backscatter_uint16(backend_context, s1_data)

    dem_data = raw_datacube_DEM(
        connection=connection,
        backend_context=backend_context,
        fetch_type=fetch_type,
    )

    # Explicitly resample DEM with bilinear interpolation
    dem_data = dem_data.resample_cube_spatial(s2_data, method="bilinear")

    # Cast DEM to UINT16
    dem_data = dem_data.linear_scale_range(0, 65534, 0, 65534)

    data = s2_data.merge_cubes(s1_data)
    data = data.merge_cubes(dem_data)

    if not disable_meteo:
        meteo_data = precomposited_datacube_METEO(
            connection=connection,
            temporal_extent=temporal_extent,
            period=composite_window,
        )

        # Explicitly resample meteo with bilinear interpolation
        meteo_data = meteo_data.resample_cube_spatial(s2_data, method="bilinear")

        data = data.merge_cubes(meteo_data)

    return data
