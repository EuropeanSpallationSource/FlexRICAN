import requests_cache
import pandas as pd
from retry_requests import retry
from typing import Optional, Sequence

import openmeteo_requests




class WeatherData:
    """Retrieve and format weather data for a single location and period."""

    def __init__(self, latitude: float, longitude: float, start_date: str, end_date: str,
                 timezone: str):
        """
        Initialize weather request settings.

        Parameters
        ----------
        latitude : float
            Latitude of the requested location.
        longitude : float
            Longitude of the requested location.
        start_date : str
            Start date in ISO format (YYYY-MM-DD).
        end_date : str
            End date in ISO format (YYYY-MM-DD).
        timezone : str
            Timezone name (e.g., "Europe/Stockholm") or "auto".
        """

        self.longitude = longitude
        self.latitude = latitude
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone

        self._openmeteo_client = None
        self._client_retries = None

    def _get_openmeteo_client(self, retries: int):
        """Build and cache the Open-Meteo client to avoid session re-creation."""
        if self._openmeteo_client is None or self._client_retries != retries:
            cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
            retry_session = retry(cache_session, retries=retries, backoff_factor=0.2)
            self._openmeteo_client = openmeteo_requests.Client(session=retry_session)
            self._client_retries = retries

        return self._openmeteo_client
    

    def _get_openmeteo_defaults(self):
        """Return default Open-Meteo hourly variables and output column mapping."""
        variables = (
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "snowfall",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
        )

        column_map = {
            "relative_humidity_2m": "relative_humidity",
            "surface_pressure": "pressure",
            "cloud_cover": "cloud_cover",
            "wind_direction_10m": "wind_direction",
            "temperature_2m": "temp_air",
            "wind_speed_10m": "wind_speed",
            "shortwave_radiation": "ghi",
            "direct_radiation": "direct_radiation",
            "diffuse_radiation": "dhi",
            "direct_normal_irradiance": "dni",
            "rain": "rain",
            "snowfall": "snowfall",
        }
        return variables, column_map

    def get_open_meteo_data(
        self,
        wind_speed_unit: str,
        variables: Optional[Sequence[str]] = None,
        retries: int = 5,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch hourly weather data from Open-Meteo archive API.

        Parameters
        ----------
        wind_speed_unit : str
            Wind speed unit accepted by Open-Meteo (e.g., "ms", "kmh", "mph").
        variables : Sequence[str] | None, optional
            Hourly variables to request. If None, a default set of variables is used.
        retries : int, optional
            Number of retry attempts for transient HTTP failures.
        verbose : bool, optional
            If True, print response metadata (coordinates, elevation, timezone).

        Returns
        -------
        pandas.DataFrame
            Timezone-aware hourly weather DataFrame indexed by datetime.
        """
        default_hourly_variables, column_map = self._get_openmeteo_defaults()
        requested_variables = list(variables) if variables is not None else list(default_hourly_variables)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "wind_speed_unit": wind_speed_unit,
            "timezone": self.timezone,
            "hourly": requested_variables,
        }

        response = self._get_openmeteo_client(retries=retries).weather_api(url, params=params)[0]

        # Process first location. Add a for-loop for multiple locations or weather models
        if verbose:
            print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation {response.Elevation()} m asl")
            print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
            print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        for index, variable in enumerate(requested_variables):
            column_map.setdefault(variable, variable)
            column_name = column_map[variable]
            hourly_data[column_name] = hourly.Variables(index).ValuesAsNumpy()

        weather = pd.DataFrame(data=hourly_data)
        weather.set_index('date', inplace=True)

        target_timezone = response.Timezone() if self.timezone == "auto" else self.timezone
        weather = weather.tz_convert(target_timezone)

        return weather

    def get_pvgis_data(
        self,
        startyear: Optional[int] = None,
        endyear: Optional[int] = None,
        usehorizon: bool = True,
        userhorizon: Optional[Sequence[float]] = None,
        url: Optional[str] = None,
        timeout: int = 30,
        coerce_year: Optional[int] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch PVGIS Typical Meteorological Year (TMY) data for pvlib ModelChain.

        Parameters
        ----------
        startyear : int | None, optional
            First year used by PVGIS to build the TMY dataset.
            If None, defaults to the year of ``self.start_date``.
        endyear : int | None, optional
            Last year used by PVGIS to build the TMY dataset.
            If None, defaults to the year of ``self.end_date``.
        usehorizon : bool, optional
            Whether to include horizon shading in PVGIS.
        userhorizon : Sequence[float] | None, optional
            Custom horizon profile in degrees for azimuth sectors.
        url : str | None, optional
            Custom PVGIS endpoint URL passed to ``pvlib.iotools.get_pvgis_tmy``.
            If None, pvlib's default URL is used.
        timeout : int, optional
            HTTP timeout in seconds for the PVGIS request.
        coerce_year : int | None, optional
            Year used by pvlib to coerce TMY timestamps. Set to None to keep
            PVGIS-selected source years instead of forcing a synthetic year
            (pvlib default is 1990).
        verbose : bool, optional
            If True, print retrieved column names.

        Returns
        -------
        pandas.DataFrame
            DataFrame named ``weather`` with a timezone-aware datetime index and
            ModelChain-ready columns including ``ghi``, ``dni``, ``dhi``,
            ``temp_air``, and ``wind_speed``.
        """
        try:
            import pvlib
        except ImportError as exc:
            raise ImportError(
                "pvlib is required for get_pvgis_data(). Install it with 'pip install pvlib'."
            ) from exc

        selected_startyear = startyear if startyear is not None else pd.Timestamp(self.start_date).year
        selected_endyear = endyear if endyear is not None else pd.Timestamp(self.end_date).year
        

        if selected_startyear >2023:
            selected_startyear = 2010

        if selected_endyear > 2023:
            selected_endyear = 2023
        print(f"Selected PVGIS years: {selected_startyear} to {selected_endyear}")
        if selected_startyear > selected_endyear:
            raise ValueError("startyear must be less than or equal to endyear.")
        
        pvgis_kwargs = {
            "startyear": selected_startyear,
            "endyear": selected_endyear,
            "usehorizon": usehorizon,
            "userhorizon": userhorizon,
            "timeout": timeout,
            "coerce_year": coerce_year,
        }
        if url is not None:
            pvgis_kwargs["url"] = url

        pvgis_result = pvlib.iotools.get_pvgis_tmy(
            self.latitude,
            self.longitude,
            **pvgis_kwargs,
        )

        weather = pvgis_result[0] if isinstance(pvgis_result, tuple) else pvgis_result

        column_aliases = {
            "G(h)": "ghi",
            "Gb(n)": "dni",
            "Gd(h)": "dhi",
            "T2m": "temp_air",
            "WS10m": "wind_speed",
            "RH": "relative_humidity",
            "SP": "pressure",
        }
        rename_map = {
            source: target
            for source, target in column_aliases.items()
            if source in weather.columns and target not in weather.columns
        }
        if rename_map:
            weather = weather.rename(columns=rename_map)

        required_columns = ("ghi", "dni", "dhi", "temp_air", "wind_speed")
        missing_required = [name for name in required_columns if name not in weather.columns]
        if missing_required:
            raise ValueError(
                "PVGIS response is missing required ModelChain columns: "
                f"{missing_required}"
            )

        if weather.index.tz is None:
            weather.index = weather.index.tz_localize("UTC")

        if self.timezone != "auto":
            weather = weather.tz_convert(self.timezone)

        if verbose:
            print(f"PVGIS columns: {list(weather.columns)}")

        return weather