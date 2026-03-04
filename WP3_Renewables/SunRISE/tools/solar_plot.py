import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pydeck as pdk
import requests
from shapely.geometry import Polygon, box


class SolarSite3DPlot:
    """Build a 3D site visualization with building roofs and solar panel polygons."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        mapbox_api_key: Optional[str] = None,
        radius: int = 500,
        zoom: int = 15,
        overpass_url: str = "http://overpass-api.de/api/interpreter",
        overpass_query_timeout: int = 8000,
        request_timeout: int = 120,
        retries: int = 5,
        backoff_factor: float = 1.5,
    ):
        """
        Initialize site and API configuration.

        Parameters
        ----------
        latitude : float
            Site latitude.
        longitude : float
            Site longitude.
        mapbox_api_key : str | None, optional
            Mapbox key used by pydeck. If None, the key is loaded from
            ``secrets.json`` (``mapbox_api_key`` field).
        radius : int, optional
            Search radius around the site in meters.
        zoom : int, optional
            Initial map zoom.
        overpass_url : str, optional
            Overpass API interpreter URL.
        overpass_query_timeout : int, optional
            Overpass server-side query timeout used in the query text.
        request_timeout : int, optional
            Client-side HTTP timeout in seconds.
        retries : int, optional
            Number of retry attempts for retriable errors.
        backoff_factor : float, optional
            Exponential backoff factor between retries.
        """
        if retries < 1:
            raise ValueError("retries must be at least 1.")

        self.latitude = latitude
        self.longitude = longitude
        self.mapbox_api_key = self._resolve_mapbox_api_key(mapbox_api_key)
        self.radius = radius
        self.zoom = zoom

        self.overpass_url = overpass_url
        self.overpass_query_timeout = overpass_query_timeout
        self.request_timeout = request_timeout
        self.retries = retries
        self.backoff_factor = backoff_factor

    def _resolve_mapbox_api_key(self, mapbox_api_key: Optional[str]) -> str:
        """
        Resolve the Mapbox API key from argument or ``secrets.json``.

        Resolution order:
        1. ``mapbox_api_key`` argument when not empty.
        2. ``secrets.json`` in current working directory.
        3. ``secrets.json`` next to this module.
        4. ``secrets.json`` in the parent folder of this module.
        """
        if isinstance(mapbox_api_key, str) and mapbox_api_key.strip():
            return mapbox_api_key.strip()

        candidate_paths = (
            Path.cwd() / "secrets.json",
            Path(__file__).resolve().parent / "secrets.json",
            Path(__file__).resolve().parent.parent / "secrets.json",
        )

        found_secrets_file = False
        for secrets_path in candidate_paths:
            if not secrets_path.exists():
                continue

            found_secrets_file = True
            try:
                with secrets_path.open("r", encoding="utf-8") as secrets_file:
                    secrets_data = json.load(secrets_file)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in '{secrets_path}'. Please fix the file and add your API key."
                ) from exc

            secret_key = secrets_data.get("mapbox_api_key")
            if isinstance(secret_key, str) and secret_key.strip():
                return secret_key.strip()

        if found_secrets_file:
            raise ValueError(
                "Mapbox API key is missing in secrets.json. "
                "Please add your API key as 'mapbox_api_key'."
            )

        raise ValueError(
            "Mapbox API key not provided and secrets.json was not found. "
            "Please add your API key to secrets.json as 'mapbox_api_key' "
            "or pass mapbox_api_key explicitly."
        )

    def _build_overpass_query(self, radius: int) -> str:
        """Create an Overpass query for buildings around the configured site."""
        return f"""
[out:json][timeout:{self.overpass_query_timeout}];
(
  way["building"](around:{radius},{self.latitude},{self.longitude});
  relation["building"](around:{radius},{self.latitude},{self.longitude});
);
(._;>;);
out body;
"""

    def _request_overpass_data(self, query: str) -> Dict[str, Any]:
        """Request Overpass data with retries, including HTTP 504 handling."""
        retriable_status_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(
                    self.overpass_url,
                    params={"data": query},
                    timeout=self.request_timeout,
                )

                if response.status_code == 504:
                    raise requests.HTTPError(
                        f"504 Server Error: Gateway Timeout for url: {response.url}",
                        response=response,
                    )

                response.raise_for_status()
                return response.json()

            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                can_retry = status_code in retriable_status_codes and attempt < self.retries

                if can_retry:
                    sleep_seconds = self.backoff_factor * (2 ** (attempt - 1))
                    time.sleep(sleep_seconds)
                    continue
                raise

            except requests.RequestException:
                if attempt < self.retries:
                    sleep_seconds = self.backoff_factor * (2 ** (attempt - 1))
                    time.sleep(sleep_seconds)
                    continue
                raise

        raise RuntimeError("Failed to get a response from Overpass API after retries.")

    def _extract_buildings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Overpass JSON data into a list of building polygon dictionaries."""
        elements = data.get("elements", [])

        node_dict = {
            el["id"]: (el["lat"], el["lon"])
            for el in elements
            if el.get("type") == "node" and "lat" in el and "lon" in el
        }

        buildings: List[Dict[str, Any]] = []
        for el in elements:
            tags = el.get("tags", {})
            if el.get("type") != "way" or "building" not in tags:
                continue

            nodes = el.get("nodes", [])
            coords = []
            all_present = True
            for node_id in nodes:
                node_coordinates = node_dict.get(node_id)
                if node_coordinates is None:
                    all_present = False
                    break

                lat, lon = node_coordinates
                coords.append((lon, lat))

            if not all_present or len(coords) <= 2:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            levels_raw = tags.get("building:levels", "1")
            try:
                levels = int(float(levels_raw))
            except (TypeError, ValueError):
                levels = 1

            elevation = max(1, levels) * 3
            buildings.append(
                {
                    "coordinates": [coords],
                    "elevation": elevation,
                    "name": tags.get("name", "N/A"),
                }
            )

        return buildings

    @staticmethod
    def _normalize_building_names(buildings: List[Dict[str, Any]]) -> None:
        """Assign deterministic names to unnamed buildings."""
        for index, building in enumerate(buildings, start=1):
            name = building.get("name", "N/A")
            if not isinstance(name, str) or not name.strip() or "N/A" in name:
                building["name"] = f"Building_{index}"

    def add_panels_to_building(
        self,
        building: Dict[str, Any],
        panels: Optional[List[Dict[str, Any]]] = None,
        panel_w_m: float = 1.6,
        panel_h_m: float = 1.0,
        spacing_m: float = 0.2,
        panel_thickness_m: float = 0.05,
        offset_above_roof_m: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Append panel polygons that fit within a building roof polygon."""
        if panels is None:
            panels = []

        if not building or "coordinates" not in building or not building["coordinates"]:
            return panels

        roof_coords = building["coordinates"][0]
        roof_poly = Polygon(roof_coords)
        if not roof_poly.is_valid:
            roof_poly = roof_poly.buffer(0)
            if not roof_poly.is_valid or roof_poly.is_empty:
                return panels

        centroid_lat = roof_poly.centroid.y
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(centroid_lat))
        if meters_per_deg_lon <= 0:
            meters_per_deg_lon = 1e-6

        panel_w_deg = panel_w_m / meters_per_deg_lon
        panel_h_deg = panel_h_m / meters_per_deg_lat
        spacing_deg_x = spacing_m / meters_per_deg_lon
        spacing_deg_y = spacing_m / meters_per_deg_lat

        shrink_eps = min(panel_w_deg, panel_h_deg) * 0.05
        roof_poly_shrunk = roof_poly.buffer(-shrink_eps)
        if roof_poly_shrunk.is_empty:
            roof_poly_shrunk = roof_poly

        min_lon, min_lat, max_lon, max_lat = roof_poly_shrunk.bounds
        eps = 1e-12
        initial_panel_count = len(panels)

        lon = min_lon
        while lon + panel_w_deg <= max_lon + eps:
            lat = min_lat
            while lat + panel_h_deg <= max_lat + eps:
                rect = box(lon, lat, lon + panel_w_deg, lat + panel_h_deg)
                if rect.within(roof_poly_shrunk):
                    coords = list(rect.exterior.coords)
                    panels.append(
                        {
                            "coordinates": [coords],
                            "elevation": building["elevation"] + panel_thickness_m + offset_above_roof_m,
                            "name": f"{building.get('name', 'building')}_panel",
                        }
                    )
                lat += panel_h_deg + spacing_deg_y
            lon += panel_w_deg + spacing_deg_x

        if len(panels) == initial_panel_count:
            for factor in (0.75, 0.5, 0.33, 0.25):
                pw = panel_w_deg * factor
                ph = panel_h_deg * factor
                sx = spacing_deg_x * factor
                sy = spacing_deg_y * factor
                local_added = 0

                lon = min_lon
                while lon + pw <= max_lon + eps:
                    lat = min_lat
                    while lat + ph <= max_lat + eps:
                        rect = box(lon, lat, lon + pw, lat + ph)
                        if rect.within(roof_poly_shrunk):
                            coords = list(rect.exterior.coords)
                            panels.append(
                                {
                                    "coordinates": [coords],
                                    "elevation": building["elevation"] + panel_thickness_m + offset_above_roof_m,
                                    "name": f"{building.get('name', 'building')}_panel",
                                }
                            )
                            local_added += 1
                        lat += ph + sy
                    lon += pw + sx

                if local_added > 0:
                    break

        return panels

    def build_deck(
        self,
        solar_roof_names: Sequence[str],
        manual_building: Optional[Dict[str, Any]] = None,
        radius: Optional[int] = None,
        zoom: Optional[int] = None,
    ) -> pdk.Deck:
        """
        Build the pydeck Deck with buildings and solar panels.

        Parameters
        ----------
        solar_roof_names : Sequence[str]
            Building names that should receive panel polygons.
        manual_building : dict | None, optional
            Optional user-provided building polygon dictionary.
        radius : int | None, optional
            Override search radius in meters.
        zoom : int | None, optional
            Override map zoom.

        Returns
        -------
        pydeck.Deck
            Configured pydeck object ready to show.
        """
        selected_radius = self.radius if radius is None else radius
        selected_zoom = self.zoom if zoom is None else zoom

        overpass_query = self._build_overpass_query(selected_radius)
        data = self._request_overpass_data(overpass_query)

        buildings = self._extract_buildings(data)
        if manual_building is not None:
            buildings.append(manual_building)

        self._normalize_building_names(buildings)

        if isinstance(solar_roof_names, str):
            selected_roofs = {solar_roof_names}
        else:
            selected_roofs = set(solar_roof_names)

        panels: List[Dict[str, Any]] = []
        for building in buildings:
            if building.get("name") in selected_roofs:
                panels = self.add_panels_to_building(building=building, panels=panels)

        building_layer = pdk.Layer(
            "PolygonLayer",
            buildings,
            get_polygon="coordinates",
            extruded=True,
            get_elevation="elevation",
            get_fill_color=[200, 160, 200, 180],
            stroked=True,
            get_line_color=[80, 80, 80],
            pickable=True,
        )

        solar_layer = pdk.Layer(
            "PolygonLayer",
            panels,
            get_polygon="coordinates",
            extruded=True,
            get_elevation="elevation",
            get_fill_color=[30, 90, 160, 220],
            stroked=True,
            get_line_color=[10, 10, 10],
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=self.latitude,
            longitude=self.longitude,
            zoom=selected_zoom,
            pitch=45,
            bearing=0,
        )

        return pdk.Deck(
            layers=[building_layer, solar_layer],
            initial_view_state=view_state,
            map_provider="mapbox",
            map_style="mapbox://styles/mapbox/streets-v12",
            api_keys={"mapbox": self.mapbox_api_key},
            height=600,
        )

    def show(
        self,
        solar_roof_names: Sequence[str],
        manual_building: Optional[Dict[str, Any]] = None,
        radius: Optional[int] = None,
        zoom: Optional[int] = None,
    ) -> pdk.Deck:
        """Build and display the deck in a notebook environment."""
        deck = self.build_deck(
            solar_roof_names=solar_roof_names,
            manual_building=manual_building,
            radius=radius,
            zoom=zoom,
        )
        deck.show()
        return deck