"""Constants for the GeoSphere Austria integration."""

from datetime import timedelta
from typing import Final

DOMAIN: Final = "geosphere_austria"
DEFAULT_NAME: Final = "GeoSphere Austria"

# API Configuration
API_BASE_URL: Final = "https://dataset.api.hub.geosphere.at/v1"
API_TIMEOUT: Final = 30

# Model endpoints
ENDPOINT_INCA_NOWCAST: Final = "timeseries/forecast/nowcast-v1-15min-1km"  # INCA Nowcast 1km
ENDPOINT_AROME: Final = "timeseries/forecast/nwp-v1-1h-2500m"  # AROME 2.5km

# Model time limits (in hours from now)
INCA_NOWCAST_LIMIT: Final = 6  # INCA Nowcast provides 0-6h forecast
AROME_LIMIT: Final = 60  # AROME provides 0-60h forecast

# Update intervals per model
INCA_UPDATE_INTERVAL: Final = timedelta(minutes=15)  # Updates every 15 minutes
AROME_UPDATE_INTERVAL: Final = timedelta(hours=3)  # Updates every 3 hours

# Legacy constants (for backward compatibility)
API_ENDPOINT: Final = ENDPOINT_AROME
UPDATE_INTERVAL: Final = AROME_UPDATE_INTERVAL
FAST_UPDATE_INTERVAL: Final = timedelta(minutes=30)

# All available parameters
ALL_PARAMETERS: Final = [
    "t2m",  # 2m temperature
    "rh2m",  # Relative humidity
    "rr_acc",  # Total precipitation
    "rain_acc",  # Rain amount
    "snow_acc",  # Snow amount
    "snowlmt",  # Snow limit altitude
    "sy",  # Weather symbol
    "u10m",  # Wind east component
    "v10m",  # Wind north component
    "ugust",  # Wind gust east
    "vgust",  # Wind gust north
    "sp",  # Surface pressure
    "tcc",  # Cloud cover
    "grad",  # Solar radiation
    "sundur_acc",  # Sunshine duration
    "mnt2m",  # Min temperature
    "mxt2m",  # Max temperature
    "cape",  # Convective energy
    "cin",  # Convective inhibition
]

# Parameter metadata
PARAMETER_INFO: Final = {
    "t2m": {"name": "Temperature", "unit": "°C", "device_class": "temperature"},
    "rh2m": {"name": "Humidity", "unit": "%", "device_class": "humidity"},
    "rr_acc": {"name": "Precipitation", "unit": "mm", "device_class": "precipitation"},
    "rain_acc": {"name": "Rain", "unit": "mm", "device_class": "precipitation"},
    "snow_acc": {"name": "Snow", "unit": "mm", "device_class": "precipitation"},
    "snowlmt": {"name": "Snow Limit", "unit": "m", "device_class": "distance"},
    "sy": {"name": "Weather Symbol", "unit": None, "device_class": None},
    "u10m": {"name": "Wind East", "unit": "m/s", "device_class": "wind_speed"},
    "v10m": {"name": "Wind North", "unit": "m/s", "device_class": "wind_speed"},
    "ugust": {"name": "Wind Gust East", "unit": "m/s", "device_class": "wind_speed"},
    "vgust": {"name": "Wind Gust North", "unit": "m/s", "device_class": "wind_speed"},
    "sp": {"name": "Pressure", "unit": "hPa", "device_class": "atmospheric_pressure"},
    "tcc": {"name": "Cloud Coverage", "unit": "%", "device_class": None},
    "grad": {"name": "Solar Radiation", "unit": "W/m²", "device_class": "irradiance"},
    "sundur_acc": {"name": "Sunshine Duration", "unit": "min", "device_class": "duration"},
    "mnt2m": {"name": "Min Temperature", "unit": "°C", "device_class": "temperature"},
    "mxt2m": {"name": "Max Temperature", "unit": "°C", "device_class": "temperature"},
    "cape": {"name": "CAPE", "unit": "m²/s²", "device_class": None},
    "cin": {"name": "CIN", "unit": "J/kg", "device_class": None},
}

# Weather condition mapping
WEATHER_SYMBOL_MAP: Final = {
    1: "sunny",
    2: "partlycloudy",
    3: "cloudy",
    4: "cloudy",
    5: "cloudy",
    6: "rainy",
    7: "rainy",
    8: "pouring",
    9: "snowy",
    10: "snowy-rainy",
}

# Configuration
CONF_LATITUDE: Final = "latitude"
CONF_LONGITUDE: Final = "longitude"
CONF_NAME: Final = "name"