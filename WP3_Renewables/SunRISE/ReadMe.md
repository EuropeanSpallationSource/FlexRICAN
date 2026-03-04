# SunRISE Tool (based on `solar_sizing_V4_present.ipynb`)

SunRISE is a notebook-driven PV sizing and assessment workflow for research sites (the provided notebook example is ESS in Lund, Sweden).  
It combines:

- 3D site visualization and rooftop panel layout,
- weather-driven PV production simulation,
- financial KPIs (LCOE, NPV, IRR, payback),
- and environmental indicators (CO₂ avoided / payback).

<span style="color:red"><strong>IMPORTANT:</strong> You must use your own Mapbox API key. Do not rely on shared keys and do not commit personal keys to version control.</span>

---

## Dependencies

### 1) Current repo requirements file
`requirements.txt` in this folder currently contains:

- `pandas`
- `matplotlib`
- `numpy`

### 2) Additional packages used by `solar_sizing_V4_present.ipynb`

- `pvlib`
- `plotly`
- `pydeck`
- `shapely`
- `requests`
- `requests-cache`
- `retry-requests`
- `openmeteo-requests`
- `ipython` / Jupyter environment

Suggested install command:

```bash
pip install pandas matplotlib numpy pvlib plotly pydeck shapely requests requests-cache retry-requests openmeteo-requests ipython
```

---

## Model Overview

### A) Site & rooftop model (`tools/solar_plot.py`)

- Pulls building footprints from OpenStreetMap via Overpass API.
- Adds optional manual buildings.
- Places panel rectangles on selected roofs using geometric constraints.
- Renders a 3D map with `pydeck` + Mapbox.

### B) PV production model (pvlib `ModelChain`)

The notebook:

1. Defines site and installation parameters (latitude/longitude, tilt, azimuth, albedo, available surface).
2. Selects module/inverter from pvlib SAM databases (`CECMod`, `cecinverter`).
3. Sizes strings/modules/inverters from electrical limits and available roof area.
4. Builds a `PVSystem` and a `ModelChain` using:
	 - transposition: `haydavies`
	 - solar position: `nrel_numpy`
	 - airmass: `kastenyoung1989`
	 - DC model: `cec`
	 - AC model: `sandia`
	 - temperature model: `sapm`
	 - losses model: `pvwatts`

Two production scenarios are evaluated:

- **Clear-sky baseline** (direct model run with weather inputs).
- **Cloud-adjusted scenario** where GHI is reduced using cloud cover, then DNI/DHI are recomputed with `pvlib.irradiance.erbs`.

### C) Financial model (`tools/models.py` – `FinancialModels`)

Financial KPIs are computed from simulated generation, including:

- LCOE,
- NPV,
- IRR,
- simple payback,
- discounted payback,
- benefit-cost ratio.

The notebook runs this through `evaluate_from_generation_df(...)` and plots KPI and LCOE evolution results.

### D) Environmental indicators

The notebook computes avoided CO₂ trajectories by combining simulated production with grid carbon intensity profiles (clear-sky vs cloudy-sky scenarios).  
`tools/models.py` also includes an `EnvironmentalModels` class for CO₂ payback calculations using lifecycle factors.

---

## Data Sources

### 1) Weather data

- **Open-Meteo Archive API** (historical hourly weather, irradiance, cloud cover, etc.).
- **PVGIS TMY** via `pvlib.iotools.get_pvgis_tmy`.

Implemented in `tools/weather.py` through `WeatherData.get_open_meteo_data(...)` and `WeatherData.get_pvgis_data(...)`.

### 2) Geospatial building data

- **OpenStreetMap / Overpass API** for building polygons.
- Optional **manual building polygon** defined in the notebook.

### 3) PV component databases

- `pvlib.pvsystem.retrieve_sam('CECMod')` for module parameters.
- `pvlib.pvsystem.retrieve_sam('cecinverter')` for inverter parameters.

### 4) Local project data files

- `data/carbon_intensity.csv` (hourly carbon-intensity time series by country code).
- `data/emissions.txt` (panel lifecycle and grid emission factors used by environmental model utilities).

### 5) Optional external carbon data (not bundled here)

The notebook references `../E_map_data/2024` for additional carbon-intensity CSV files.  
If you use that section, make sure this folder exists and contains the expected files.

---

## Mapbox API Key Setup

Create or update `secrets.json` in `WP3_Renewables/SunRISE/`:

```json
{
	"mapbox_api_key": "YOUR_MAPBOX_PUBLIC_TOKEN"
}
```

`SolarSite3DPlot` will read this key automatically if `mapbox_api_key=None`.

<span style="color:red"><strong>NOTE:</strong> Every user must provide their own Mapbox API key.</span>

