from typing import Optional

import geopandas as gpd
from ipyleaflet import DrawControl, LayersControl, Map, SearchControl, basemaps
from IPython.display import display
from ipywidgets import HTML, widgets
from openeo_gfmap import BoundingBoxExtent
from shapely import geometry
from shapely.geometry import Polygon, shape


def handle_draw(instance, action, geo_json, output, area_limit):
    with output:
        if action == "created":
            poly = Polygon(shape(geo_json.get("geometry")))
            bbox = poly.bounds
            display(HTML(f"<b>Your extent:</b> {bbox}"))

            # We convert our bounding box to local UTM projection
            # for further processing
            bbox_utm, epsg = _latlon_to_utm(bbox)
            area = (bbox_utm[2] - bbox_utm[0]) * (bbox_utm[3] - bbox_utm[1]) / 1000000
            display(HTML(f"<b>Area of extent:</b> {area:.2f} km²"))

            if area_limit is not None:
                if area > area_limit:
                    message = f"Area of extent is too large. Please select an area smaller than {area_limit} km²."
                    display(HTML(f'<span style="color: red;"><b>{message}</b></span>'))
                    instance.last_draw = {"type": "Feature", "geometry": None}

        elif action == "deleted":
            instance.clear()
            instance.last_draw = {"type": "Feature", "geometry": None}

        else:
            raise ValueError(f"Unknown action: {action}")


class ui_map:
    def __init__(self, area_limit: Optional[int] = None):
        """
        Initializes an ipyleaflet map with a draw control to select an extent.

        Parameters
        ----------
        area_limit : int, optional
            The maximum area in km² that can be selected on the map.
            By default no restrictions are imposed.
        """
        from ipyleaflet import basemap_to_tiles

        self.output = widgets.Output()
        self.area_limit = area_limit
        osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
        osm.base = True
        osm.name = "Open street map"

        img = basemap_to_tiles(basemaps.Esri.WorldImagery)
        img.base = True
        img.name = "Satellite imagery"

        self.map = Map(
            center=(51.1872, 5.1154), zoom=2, layers=[img, osm], scroll_wheel_zoom=True
        )
        self.map.add_control(LayersControl())

        self.draw_control = DrawControl(edit=False)

        self.draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "#6be5c3",
                "color": "#00F",
                "fillOpacity": 0.3,
            },
            "drawError": {"color": "#dd253b", "message": "Oups!"},
            "allowIntersection": False,
            "metric": ["km"],
        }
        self.draw_control.circle = {}
        self.draw_control.polyline = {}
        self.draw_control.circlemarker = {}
        self.draw_control.polygon = {}

        # Wrapper to pass additional arguments
        def draw_handler(instance, action, geo_json):
            handle_draw(
                instance, action, geo_json, self.output, area_limit=self.area_limit
            )

        # Attach the event listener to the draw control
        self.draw_control.on_draw(draw_handler)

        self.map.add_control(self.draw_control)

        search = SearchControl(
            position="topleft",
            url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
            zoom=20,
        )
        self.map.add_control(search)

        self.spatial_extent = None
        self.bbox = None
        self.poly = None

        self.show_map()

    def show_map(self):
        vbox = widgets.VBox(
            [self.map, self.output],
            layout={"height": "600px"},
        )
        return display(vbox)

    def get_extent(self, projection="utm") -> BoundingBoxExtent:
        """Get extent from last drawn rectangle on the map.

        Parameters
        ----------
        projection : str, optional
            The projection to use for the extent.
            You can either request "latlon" or "utm". In case of the latter, the
            local utm projection is automatically derived.

        Returns
        -------
        BoundingBoxExtent
            The extent as a bounding box in the requested projection.

        Raises
        ------
        ValueError
            If no rectangle has been drawn on the map.
        """

        obj = self.draw_control.last_draw

        if obj.get("geometry") is None:
            raise ValueError(
                "Please first draw a rectangle on the map before proceeding."
            )

        self.poly = Polygon(shape(obj.get("geometry")))
        if self.poly is None:
            return None

        bbox = self.poly.bounds

        if projection == "utm":
            bbox_utm, epsg = _latlon_to_utm(bbox)
            self.spatial_extent = BoundingBoxExtent(*bbox_utm, epsg)
        else:
            self.spatial_extent = BoundingBoxExtent(*bbox)

        return self.spatial_extent

    def get_polygon_latlon(self):
        self.get_extent()
        return self.poly


def _latlon_to_utm(bbox):
    """This function converts a bounding box defined in lat/lon
    to local UTM coordinates.
    It returns the bounding box in UTM and the epsg code
    of the resulting UTM projection."""

    # convert bounding box to geodataframe
    bbox_poly = geometry.box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")

    # estimate best UTM zone
    crs = bbox_gdf.estimate_utm_crs()
    epsg = int(crs.to_epsg())

    # convert to UTM
    bbox_utm = bbox_gdf.to_crs(crs).total_bounds

    return bbox_utm, epsg
