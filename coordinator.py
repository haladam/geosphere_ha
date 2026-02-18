"""DataUpdateCoordinator for GeoSphere Austria with INCA Nowcast + AROME."""

from __future__ import annotations
from datetime import datetime, timedelta
import logging
import math
import aiohttp
import async_timeout

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    API_BASE_URL,
    API_TIMEOUT,
    DOMAIN,
    ENDPOINT_INCA_NOWCAST,
    ENDPOINT_AROME,
    INCA_NOWCAST_LIMIT,
    INCA_UPDATE_INTERVAL,
    AROME_UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)


class GeoSphereDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching GeoSphere data from INCA Nowcast and AROME models."""

    def __init__(
        self,
        hass: HomeAssistant,
        session: aiohttp.ClientSession,
        latitude: float,
        longitude: float,
    ) -> None:
        """Initialize."""
        self.latitude = latitude
        self.longitude = longitude
        self.session = session

        # Store data per model
        self._raw_forecast_data_inca = None
        self._raw_forecast_data_arome = None

        # Track last API update per model
        self._last_api_update_inca = None
        self._last_api_update_arome = None

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),  # Check every 5 minutes for INCA updates
        )

    async def _async_update_data(self) -> dict:
        """Fetch data from APIs or update current values from cached forecasts."""
        now = dt_util.utcnow()

        # Determine which models need fresh data
        should_fetch_inca = (
            self._last_api_update_inca is None
            or now - self._last_api_update_inca >= INCA_UPDATE_INTERVAL
        )
        should_fetch_arome = (
            self._last_api_update_arome is None
            or now - self._last_api_update_arome >= AROME_UPDATE_INTERVAL
        )

        # Fetch data from each model as needed
        if should_fetch_inca:
            _LOGGER.info("Fetching fresh data from INCA Nowcast")
            # Based on user confirmation, INCA provides: rr, dd, fx, pt, rh2m, t2m, td
            # We request: t2m, rh2m, rr (precip), dd (wind dir), fx (gusts)
            self._raw_forecast_data_inca = await self._fetch_from_api(
                ENDPOINT_INCA_NOWCAST,
                "t2m,rh2m,rr,dd,fx"
            )
            if self._raw_forecast_data_inca:
                self._last_api_update_inca = now

        if should_fetch_arome:
            _LOGGER.info("Fetching fresh data from AROME")
            self._raw_forecast_data_arome = await self._fetch_from_api(
                ENDPOINT_AROME,
                "t2m,rh2m,rr_acc,rain_acc,snow_acc,sy,u10m,v10m,ugust,vgust,grad,tcc,sp,snowlmt,sundur_acc,mnt2m,mxt2m,cape,cin"
            )
            if self._raw_forecast_data_arome:
                self._last_api_update_arome = now

        # Merge forecasts from both models
        if not any([self._raw_forecast_data_inca, self._raw_forecast_data_arome]):
            raise UpdateFailed("No forecast data available from any model")

        return self._merge_forecasts(now)

    async def _fetch_from_api(self, endpoint: str, parameters: str) -> dict | None:
        """Fetch data from specific API endpoint with model-specific parameters."""
        url = f"{API_BASE_URL}/{endpoint}"
        params = {
            "lat_lon": f"{self.latitude},{self.longitude}",
            "parameters": parameters,
        }

        _LOGGER.debug("Requesting data from %s with parameters: %s", url, parameters)
        try:
            async with async_timeout.timeout(API_TIMEOUT):
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        _LOGGER.warning("API error %s for %s: %s", response.status, endpoint, response_text)
                        return None

                    import json
                    response_text = await response.text()
                    data = json.loads(response_text)
                    return data

        except aiohttp.ClientError as err:
            _LOGGER.warning("Error communicating with API %s: %s", endpoint, err)
            return None
        except Exception as err:
            _LOGGER.warning("Unexpected error fetching data from %s: %s", endpoint, err)
            return None

    def _merge_forecasts(self, current_time: datetime) -> dict:
        """Merge forecasts from INCA Nowcast and AROME based on time horizons."""
        merged_forecast = []
        timestamps = []

        # Process each model's data
        inca_data = self._process_model_data(self._raw_forecast_data_inca, current_time, "INCA-Nowcast") if self._raw_forecast_data_inca else None
        arome_data = self._process_model_data(self._raw_forecast_data_arome, current_time, "AROME") if self._raw_forecast_data_arome else None

        # Determine current conditions - merge INCA + AROME parameters
        current = None
        current_index = 0
        current_timestamp = None

        if inca_data and arome_data:
            # Best case: merge both
            current = self._merge_parameters(inca_data["current"], arome_data["current"])
            current_index = inca_data["current_index"]
            current_timestamp = inca_data["current_timestamp"]
        elif inca_data:
            current = inca_data["current"]
            current_index = inca_data["current_index"]
            current_timestamp = inca_data["current_timestamp"]
        elif arome_data:
            current = arome_data["current"]
            current_index = arome_data["current_index"]
            current_timestamp = arome_data["current_timestamp"]

        if current is None:
            raise UpdateFailed("No current conditions available from any model")

        # Build merged forecast by time horizon
        max_hours = 60

        for i in range(max_hours * 4):  # Process in 15-minute increments
            forecast_time = current_time + timedelta(minutes=i * 15)
            hours_ahead = i / 4

            forecast_point = None

            # Select and merge sources based on time horizon
            if hours_ahead < INCA_NOWCAST_LIMIT:
                # 0-6 hours: Use INCA for high-res data, supplement with AROME
                if inca_data and i < len(inca_data["forecast"]):
                    inca_point = inca_data["forecast"][i].copy()

                    # Find matching AROME hourly data
                    if arome_data:
                        # Find the closest AROME hour (AROME is hourly)
                        arome_idx = int(hours_ahead)
                        if arome_idx < len(arome_data["forecast"]):
                            arome_point = arome_data["forecast"][arome_idx]
                            # Merge: INCA parameters take priority, AROME fills gaps
                            forecast_point = self._merge_parameters(inca_point, arome_point)
                            forecast_point["source"] = "INCA-Nowcast+AROME"
                        else:
                            forecast_point = inca_point
                            forecast_point["source"] = "INCA-Nowcast"
                    else:
                        forecast_point = inca_point
                        forecast_point["source"] = "INCA-Nowcast"

                elif arome_data and i % 4 == 0:  # INCA not available, fall back to AROME
                    hour_index = int(hours_ahead)
                    if hour_index < len(arome_data["forecast"]):
                        forecast_point = arome_data["forecast"][hour_index].copy()
                        forecast_point["source"] = "AROME"
            else:
                # 6+ hours: Use AROME only (hourly resolution)
                if arome_data and i % 4 == 0:  # On the hour
                    hour_index = int(hours_ahead)
                    if hour_index < len(arome_data["forecast"]):
                        forecast_point = arome_data["forecast"][hour_index].copy()
                        forecast_point["source"] = "AROME"

            if forecast_point:
                if "datetime" not in forecast_point or forecast_point["datetime"] is None:
                    forecast_point["datetime"] = forecast_time.isoformat()

                merged_forecast.append(forecast_point)
                timestamps.append(forecast_time.isoformat())

        _LOGGER.info(
            "Merged forecast: %d points (INCA: %s, AROME: %s)",
            len(merged_forecast),
            "available" if inca_data else "unavailable",
            "available" if arome_data else "unavailable"
        )

        return {
            "reference_time": current_time.isoformat(),
            "current": current,
            "current_index": current_index,
            "current_timestamp": current_timestamp,
            "forecast": merged_forecast,
            "timestamps": timestamps,
            "last_api_update_inca": self._last_api_update_inca.isoformat() if self._last_api_update_inca else None,
            "last_api_update_arome": self._last_api_update_arome.isoformat() if self._last_api_update_arome else None,
        }

    def _merge_parameters(self, inca_data: dict, arome_data: dict) -> dict:
        """Merge parameters with INCA taking priority for overlapping parameters.

        INCA provides: t2m, rh2m, rr, dd, fx (high-resolution)
        AROME provides: All parameters including wind u/v, sy, cape, etc.
        """
        merged = arome_data.copy()  # Start with AROME

        # Override with INCA data ONLY for what INCA actually provides
        # Include 'wind_bearing' (from dd) and 'wind_gust_speed' (from fx)
        inca_priority_params = ['t2m', 'rh2m', 'rr', 'rr_hourly', 'wind_bearing', 'wind_gust_speed']

        for param in inca_priority_params:
            if param in inca_data and inca_data[param] is not None:
                merged[param] = inca_data[param]

        return merged

    def _process_model_data(self, raw_data: dict, current_time: datetime, model_name: str) -> dict | None:
        """Process raw API data from a single model."""
        try:
            if "features" in raw_data:
                timestamps = raw_data.get("timestamps", [])
                parameters = raw_data["features"][0]["properties"]["parameters"]
            else:
                timestamps = raw_data.get("timestamps", [])
                parameters = {}
                for key, value in raw_data.items():
                    if key == "timestamps":
                        continue
                    if isinstance(value, dict) and "data" in value:
                        parameters[key] = value
                    elif isinstance(value, list):
                        parameters[key] = {"data": value}

            if not parameters or not timestamps:
                _LOGGER.warning("%s: No parameters or timestamps found", model_name)
                return None

            # Calculate hourly values
            precip_hourly = []
            if "rr_acc" in parameters:
                precip_hourly = self._calculate_hourly_values(parameters.get("rr_acc", {}).get("data", []))
            elif "rr" in parameters:
                precip_data = parameters.get("rr", {}).get("data", [])
                # INCA rr is accumulation for the 15-min interval, not instantaneous rate
                precip_hourly = [max(0, val) if val is not None else 0 for val in precip_data]

            rain_hourly = self._calculate_hourly_values(parameters.get("rain_acc", {}).get("data", []))
            snow_hourly = self._calculate_hourly_values(parameters.get("snow_acc", {}).get("data", []))
            solar_hourly = self._calculate_hourly_values(parameters.get("grad", {}).get("data", []))
            sunshine_hourly = self._calculate_hourly_values(parameters.get("sundur_acc", {}).get("data", []))

            current_index = self._find_current_index(timestamps, current_time)

            current = {}
            for param_key, param_data in parameters.items():
                data_array = param_data.get("data", []) if isinstance(param_data, dict) else param_data
                if data_array and current_index < len(data_array):
                    current[param_key] = data_array[current_index]
                else:
                    current[param_key] = None

            current["rr_hourly"] = precip_hourly[current_index] if current_index < len(precip_hourly) else 0
            current["rain_hourly"] = rain_hourly[current_index] if current_index < len(rain_hourly) else 0
            current["snow_hourly"] = snow_hourly[current_index] if current_index < len(snow_hourly) else 0
            current["grad_hourly"] = solar_hourly[current_index] if current_index < len(solar_hourly) else 0
            current["sundur_hourly"] = sunshine_hourly[current_index] if current_index < len(sunshine_hourly) else 0

            # Calculate wind from u/v OR use direct dd/ff/fx
            u = current.get("u10m")
            v = current.get("v10m")
            if u is not None and v is not None:
                current["wind_speed"] = self._calculate_wind_speed(u, v)
                current["wind_bearing"] = self._calculate_wind_bearing(u, v)
            elif "dd" in current: # Direct wind direction from INCA
                current["wind_bearing"] = current["dd"]
                # If ff (speed) is missing in INCA, we can't calculate speed easily without u/v or ff
                # But user JSON shows 'dd' and 'fx' (gust), not 'ff' (avg speed).
                # We'll set wind_speed to 0 or use gust as proxy if critically needed, but better to leave 0.
                current["wind_speed"] = 0 

            # Handle gusts
            ugust = current.get("ugust")
            vgust = current.get("vgust")
            if ugust is not None and vgust is not None:
                current["wind_gust_speed"] = self._calculate_wind_speed(ugust, vgust)
            elif "fx" in current: # Direct gust speed from INCA
                current["wind_gust_speed"] = current["fx"]
            else:
                current["wind_gust_speed"] = 0

            forecast = []
            for i in range(len(timestamps)):
                forecast_point = {"datetime": timestamps[i]}

                for param_key, param_data in parameters.items():
                    data_array = param_data.get("data", []) if isinstance(param_data, dict) else param_data
                    if data_array and i < len(data_array):
                        forecast_point[param_key] = data_array[i]
                    else:
                        forecast_point[param_key] = None

                forecast_point["rr_hourly"] = precip_hourly[i] if i < len(precip_hourly) else 0
                forecast_point["rain_hourly"] = rain_hourly[i] if i < len(rain_hourly) else 0
                forecast_point["snow_hourly"] = snow_hourly[i] if i < len(snow_hourly) else 0
                forecast_point["grad_hourly"] = solar_hourly[i] if i < len(solar_hourly) else 0
                forecast_point["sundur_hourly"] = sunshine_hourly[i] if i < len(sunshine_hourly) else 0

                # Wind calculation for forecast points
                u = forecast_point.get("u10m")
                v = forecast_point.get("v10m")
                if u is not None and v is not None:
                    forecast_point["wind_speed"] = self._calculate_wind_speed(u, v)
                    forecast_point["wind_bearing"] = self._calculate_wind_bearing(u, v)
                elif "dd" in forecast_point:
                     forecast_point["wind_bearing"] = forecast_point["dd"]
                     forecast_point["wind_speed"] = 0 # Missing 'ff' in INCA output

                # Gusts for forecast points
                ugust = forecast_point.get("ugust")
                vgust = forecast_point.get("vgust")
                if ugust is not None and vgust is not None:
                    forecast_point["wind_gust_speed"] = self._calculate_wind_speed(ugust, vgust)
                elif "fx" in forecast_point:
                    forecast_point["wind_gust_speed"] = forecast_point["fx"]

                forecast.append(forecast_point)

            _LOGGER.debug("%s: Processed %d forecast points", model_name, len(forecast))

            return {
                "current": current,
                "current_index": current_index,
                "current_timestamp": timestamps[current_index] if current_index < len(timestamps) else None,
                "forecast": forecast,
                "timestamps": timestamps,
                "parameters": parameters,
            }

        except (KeyError, IndexError, TypeError) as err:
            _LOGGER.warning("Error processing %s data: %s", model_name, err)
            return None

    @staticmethod
    def _find_current_index(timestamps: list, current_time: datetime) -> int:
        """Find the forecast index closest to the current time."""
        if not timestamps:
            return 0

        min_diff = None
        closest_index = 0

        for i, ts_str in enumerate(timestamps):
            try:
                ts = datetime.fromisoformat(ts_str.replace("+00:00", "+00:00"))
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=ts.tzinfo)

                diff = abs((ts - current_time).total_seconds())
                if ts >= current_time or diff <= 1800:  # 30 minutes
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_index = i
            except (ValueError, AttributeError) as err:
                _LOGGER.debug("Error parsing timestamp %s: %s", ts_str, err)
                continue

        return closest_index

    @staticmethod
    def _calculate_hourly_values(accumulated: list) -> list:
        """Calculate hourly values from accumulated values."""
        if not accumulated or len(accumulated) == 0:
            return []

        hourly = []
        for i in range(len(accumulated)):
            if i == 0:
                hourly.append(max(0, accumulated[i]))
            else:
                diff = accumulated[i] - accumulated[i - 1]
                hourly.append(max(0, diff))

        return hourly

    @staticmethod
    def _calculate_wind_speed(u: float, v: float) -> float:
        """Calculate wind speed from U and V components."""
        if u is None or v is None:
            return 0
        return round(math.sqrt(u**2 + v**2), 1)

    @staticmethod
    def _calculate_wind_bearing(u: float, v: float) -> float:
        """Calculate wind bearing from U and V components."""
        if u is None or v is None or (u == 0 and v == 0):
            return 0
        bearing = (270 - math.atan2(v, u) * 180 / math.pi) % 360
        return round(bearing)
