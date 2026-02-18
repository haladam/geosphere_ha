"""Weather platform for GeoSphere Austria."""

from __future__ import annotations
from datetime import datetime
import logging
from typing import Any
from collections import defaultdict

from homeassistant.components.weather import (
    Forecast,
    WeatherEntity,
    WeatherEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfPrecipitationDepth,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, WEATHER_SYMBOL_MAP
from .coordinator import GeoSphereDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up GeoSphere weather entity."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([GeoSphereWeather(coordinator, entry)])


class GeoSphereWeather(CoordinatorEntity, WeatherEntity):
    """Implementation of GeoSphere Austria weather entity."""

    _attr_native_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_native_precipitation_unit = UnitOfPrecipitationDepth.MILLIMETERS
    _attr_native_pressure_unit = UnitOfPressure.HPA
    _attr_native_wind_speed_unit = UnitOfSpeed.KILOMETERS_PER_HOUR
    _attr_supported_features = (
        WeatherEntityFeature.FORECAST_HOURLY | WeatherEntityFeature.FORECAST_DAILY
    )

    def __init__(
        self,
        coordinator: GeoSphereDataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the weather entity."""
        super().__init__(coordinator)
        self._attr_name = entry.data["name"]
        self._attr_unique_id = f"{entry.entry_id}_weather"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.data["name"],
            "manufacturer": "GeoSphere Austria",
            "model": "Multi-Model (INCA Nowcast 1km + AROME 2.5km)",
            "entry_type": "service",
        }

    @property
    def native_temperature(self) -> float | None:
        """Return the temperature."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data["current"].get("t2m")

    @property
    def humidity(self) -> float | None:
        """Return the humidity."""
        if not self.coordinator.data:
            return None
        humidity = self.coordinator.data["current"].get("rh2m")
        return round(humidity) if humidity is not None else None

    @property
    def native_pressure(self) -> float | None:
        """Return the pressure in hPa."""
        if not self.coordinator.data:
            return None
        pressure = self.coordinator.data["current"].get("sp")
        return round(pressure / 100, 1) if pressure else None

    @property
    def native_wind_speed(self) -> float | None:
        """Return the wind speed in km/h."""
        if not self.coordinator.data:
            return None
        wind_ms = self.coordinator.data["current"].get("wind_speed")
        return round(wind_ms * 3.6, 1) if wind_ms else None

    @property
    def wind_bearing(self) -> float | None:
        """Return the wind bearing."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data["current"].get("wind_bearing")

    @property
    def cloud_coverage(self) -> float | None:
        """Return the cloud coverage."""
        if not self.coordinator.data:
            return None
        tcc = self.coordinator.data["current"].get("tcc")
        return round(tcc * 100) if tcc is not None else None

    @property
    def condition(self) -> str | None:
        """Return the current condition."""
        if not self.coordinator.data:
            return None
        symbol = self.coordinator.data["current"].get("sy")
        if symbol is not None:
            return WEATHER_SYMBOL_MAP.get(int(symbol), "cloudy")
        return None

    @property
    def native_wind_gust_speed(self) -> float | None:
        """Return the wind gust speed in km/h."""
        if not self.coordinator.data:
            return None
        gust_ms = self.coordinator.data["current"].get("wind_gust_speed")
        return round(gust_ms * 3.6, 1) if gust_ms else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        if not self.coordinator.data:
            return {}

        current = self.coordinator.data["current"]
        attributes = {
            "attribution": "Data provided by GeoSphere Austria",
            "reference_time": self.coordinator.data.get("reference_time"),
            "current_timestamp": self.coordinator.data.get("current_timestamp"),
            "forecast_hours_available": len(self.coordinator.data.get("forecast", [])),
        }

        # Show which model provided current data and last update times
        current_forecast = self.coordinator.data.get("forecast", [])
        if current_forecast and len(current_forecast) > 0:
            attributes["current_model"] = current_forecast[0].get("source", "AROME")

        # Show last update times for each model
        if self.coordinator.data.get("last_api_update_inca"):
            attributes["inca_nowcast_last_update"] = self.coordinator.data["last_api_update_inca"]
        if self.coordinator.data.get("last_api_update_arome"):
            attributes["arome_last_update"] = self.coordinator.data["last_api_update_arome"]

        # Add additional useful attributes
        if current.get("snowlmt"):
            attributes["snow_limit"] = round(current["snowlmt"])
        if current.get("mnt2m") is not None:
            attributes["temp_min"] = round(current["mnt2m"], 1)
        if current.get("mxt2m") is not None:
            attributes["temp_max"] = round(current["mxt2m"], 1)
        if current.get("cape") is not None:
            attributes["cape"] = round(current["cape"], 1)

        # Use hourly values instead of accumulated
        if current.get("grad_hourly"):
            attributes["solar_radiation_hourly"] = round(current["grad_hourly"] / 3600)  # W/m²
            attributes["uv_index"] = round((current["grad_hourly"] / 3600) / 100, 1)
        if current.get("sundur_hourly") is not None:
            attributes["sunshine_duration_minutes"] = round(current["sundur_hourly"] / 60, 1)
        if current.get("rain_hourly") is not None:
            attributes["rain_hourly"] = round(current["rain_hourly"], 2)
        if current.get("snow_hourly") is not None:
            attributes["snow_hourly"] = round(current["snow_hourly"], 2)

        return attributes

    async def async_forecast_hourly(self) -> list[Forecast] | None:
        """Return the hourly forecast with 15-min INCA data aggregated to hourly."""
        if not self.coordinator.data or not self.coordinator.data.get("forecast"):
            return None

        forecasts = []
        forecast_data = self.coordinator.data["forecast"]

        # Group forecast points by hour for aggregation
        hourly_groups = defaultdict(list)

        for forecast_point in forecast_data:
            try:
                dt_str = forecast_point.get("datetime")
                if not dt_str:
                    continue

                dt = datetime.fromisoformat(dt_str.replace("+00:00", "+00:00"))

                # Round down to the hour to group 15-min intervals
                hour_key = dt.replace(minute=0, second=0, microsecond=0).isoformat()

                hourly_groups[hour_key].append(forecast_point)

            except (ValueError, TypeError) as err:
                _LOGGER.debug("Error parsing forecast timestamp: %s", err)
                continue

        # Process each hourly group
        for hour_key in sorted(hourly_groups.keys()):
            points = hourly_groups[hour_key]

            if not points:
                continue

            # Aggregate the data from multiple 15-min intervals
            aggregated = self._aggregate_forecast_points(points)

            if not aggregated:
                continue

            # Determine condition
            condition = None
            if aggregated.get("sy") is not None:
                condition = WEATHER_SYMBOL_MAP.get(int(aggregated["sy"]), "cloudy")

            # Convert values
            pressure = aggregated.get("sp")
            pressure_hpa = round(pressure / 100, 1) if pressure else None

            tcc = aggregated.get("tcc")
            cloud_coverage = round(tcc * 100) if tcc is not None else None

            wind_ms = aggregated.get("wind_speed")
            wind_kmh = round(wind_ms * 3.6, 1) if wind_ms else None

            wind_gust_ms = aggregated.get("wind_gust_speed")
            wind_gust_kmh = round(wind_gust_ms * 3.6, 1) if wind_gust_ms else None

            # UV Index from hourly solar radiation
            grad_hourly = aggregated.get("grad_hourly")
            uv_index = round((grad_hourly / 3600) / 100, 1) if grad_hourly else None

            # Use aggregated precipitation
            precip_hourly = aggregated.get("rr_hourly", 0)

            forecast = Forecast(
                datetime=hour_key,
                native_temperature=aggregated.get("t2m"),
                native_templow=aggregated.get("mnt2m"),
                native_apparent_temperature=aggregated.get("mxt2m"),
                humidity=round(aggregated["rh2m"]) if aggregated.get("rh2m") else None,
                native_precipitation=round(precip_hourly, 2) if precip_hourly else 0,
                native_pressure=pressure_hpa,
                native_wind_speed=wind_kmh,
                native_wind_gust_speed=wind_gust_kmh,
                wind_bearing=aggregated.get("wind_bearing"),
                cloud_coverage=cloud_coverage,
                condition=condition,
                uv_index=uv_index,
            )

            forecasts.append(forecast)

        _LOGGER.info("Created %s hourly forecast points (aggregated from 15-min data)", len(forecasts))
        return forecasts if forecasts else None

    def _aggregate_forecast_points(self, points: list[dict]) -> dict:
        """Aggregate multiple 15-minute forecast points into a single hourly forecast."""
        if not points:
            return {}

        if len(points) == 1:
            return points[0]

        aggregated = {}

        # Temperature - average
        temps = [p.get("t2m") for p in points if p.get("t2m") is not None]
        if temps:
            aggregated["t2m"] = round(sum(temps) / len(temps), 1)

        # Humidity - average
        humidities = [p.get("rh2m") for p in points if p.get("rh2m") is not None]
        if humidities:
            aggregated["rh2m"] = round(sum(humidities) / len(humidities), 1)

        # Precipitation - sum all 15-min intervals
        precips = [p.get("rr_hourly", 0) for p in points]
        aggregated["rr_hourly"] = sum(precips)

        # Wind speed - average
        wind_speeds = [p.get("wind_speed") for p in points if p.get("wind_speed") is not None]
        if wind_speeds:
            aggregated["wind_speed"] = round(sum(wind_speeds) / len(wind_speeds), 1)

        # Wind bearing - take most recent (last point)
        for p in reversed(points):
            if p.get("wind_bearing") is not None:
                aggregated["wind_bearing"] = p["wind_bearing"]
                break

        # Wind gusts - maximum
        gusts = [p.get("wind_gust_speed") for p in points if p.get("wind_gust_speed") is not None]
        if gusts:
            aggregated["wind_gust_speed"] = round(max(gusts), 1)

        # Pressure - average
        pressures = [p.get("sp") for p in points if p.get("sp") is not None]
        if pressures:
            aggregated["sp"] = round(sum(pressures) / len(pressures), 1)

        # Cloud cover - average
        clouds = [p.get("tcc") for p in points if p.get("tcc") is not None]
        if clouds:
            aggregated["tcc"] = round(sum(clouds) / len(clouds), 3)

        # Weather symbol - most severe
        symbols = [p.get("sy") for p in points if p.get("sy") is not None]
        if symbols:
            aggregated["sy"] = max(symbols)

        # Solar radiation - sum (total energy over the hour)
        solar = [p.get("grad_hourly", 0) for p in points]
        aggregated["grad_hourly"] = sum(solar)

        # Min/Max temps - take from AROME if available
        for p in points:
            if p.get("mnt2m") is not None and "mnt2m" not in aggregated:
                aggregated["mnt2m"] = p["mnt2m"]
            if p.get("mxt2m") is not None and "mxt2m" not in aggregated:
                aggregated["mxt2m"] = p["mxt2m"]

        return aggregated

    async def async_forecast_daily(self) -> list[Forecast] | None:
        """Return the daily forecast aggregated from hourly data."""
        if not self.coordinator.data or not self.coordinator.data.get("forecast"):
            return None

        # Group hourly data by day
        daily_data = defaultdict(lambda: {
            'temp_min': float('inf'),
            'temp_max': float('-inf'),
            'humidity_sum': 0,
            'humidity_count': 0,
            'precip_sum': 0,
            'wind_sum': 0,
            'wind_count': 0,
            'gust_max': 0,
            'pressure_sum': 0,
            'pressure_count': 0,
            'cloud_sum': 0,
            'cloud_count': 0,
            'uv_max': 0,
            'conditions': [],
            'datetime': None,
        })

        forecast_data = self.coordinator.data["forecast"]

        for forecast_point in forecast_data:
            try:
                dt_str = forecast_point.get("datetime")
                if not dt_str:
                    continue

                dt = datetime.fromisoformat(dt_str.replace("+00:00", "+00:00"))
                day = dt.date().isoformat()

                # Store first datetime at noon for the day
                if daily_data[day]['datetime'] is None:
                    daily_data[day]['datetime'] = f"{day}T12:00:00+00:00"

                # Temperature
                temp = forecast_point.get("t2m")
                if temp is not None:
                    daily_data[day]['temp_min'] = min(daily_data[day]['temp_min'], temp)
                    daily_data[day]['temp_max'] = max(daily_data[day]['temp_max'], temp)

                # Humidity
                humidity = forecast_point.get("rh2m")
                if humidity is not None:
                    daily_data[day]['humidity_sum'] += humidity
                    daily_data[day]['humidity_count'] += 1

                # Precipitation
                precip_hourly = forecast_point.get("rr_hourly", 0)
                if precip_hourly is not None:
                    daily_data[day]['precip_sum'] += precip_hourly

                # Wind
                wind_ms = forecast_point.get("wind_speed")
                if wind_ms is not None:
                    daily_data[day]['wind_sum'] += wind_ms * 3.6
                    daily_data[day]['wind_count'] += 1

                # Wind gust
                gust_ms = forecast_point.get("wind_gust_speed")
                if gust_ms is not None:
                    gust_kmh = gust_ms * 3.6
                    daily_data[day]['gust_max'] = max(daily_data[day]['gust_max'], gust_kmh)

                # Pressure
                pressure = forecast_point.get("sp")
                if pressure is not None:
                    daily_data[day]['pressure_sum'] += pressure / 100
                    daily_data[day]['pressure_count'] += 1

                # Cloud coverage
                tcc = forecast_point.get("tcc")
                if tcc is not None:
                    daily_data[day]['cloud_sum'] += tcc * 100
                    daily_data[day]['cloud_count'] += 1

                # UV Index
                grad_hourly = forecast_point.get("grad_hourly")
                if grad_hourly is not None:
                    uv = (grad_hourly / 3600) / 100
                    daily_data[day]['uv_max'] = max(daily_data[day]['uv_max'], uv)

                # Conditions during daytime
                if 6 <= dt.hour <= 20 and forecast_point.get("sy") is not None:
                    condition_code = int(forecast_point["sy"])
                    daily_data[day]['conditions'].append({
                        'code': condition_code,
                        'hour': dt.hour,
                        'precip': forecast_point.get("rr_hourly", 0)
                    })

            except (KeyError, ValueError, TypeError) as err:
                _LOGGER.debug("Error processing daily forecast point: %s", err)
                continue

        # Build daily forecasts
        forecasts = []
        for day in sorted(daily_data.keys()):
            data = daily_data[day]

            if data['humidity_count'] == 0:
                continue

            # Calculate averages
            avg_humidity = round(data['humidity_sum'] / data['humidity_count'])
            avg_wind = round(data['wind_sum'] / data['wind_count'], 1) if data['wind_count'] > 0 else None
            avg_pressure = round(data['pressure_sum'] / data['pressure_count'], 1) if data['pressure_count'] > 0 else None
            avg_cloud = round(data['cloud_sum'] / data['cloud_count']) if data['cloud_count'] > 0 else None

            # Total precipitation
            total_precip = round(data['precip_sum'], 1)

            # Determine condition
            condition = self._select_daily_condition(data['conditions'])

            forecast = Forecast(
                datetime=data['datetime'],
                native_temperature=round(data['temp_max'], 1) if data['temp_max'] != float('-inf') else None,
                native_templow=round(data['temp_min'], 1) if data['temp_min'] != float('inf') else None,
                humidity=avg_humidity,
                native_precipitation=total_precip,
                native_pressure=avg_pressure,
                native_wind_speed=avg_wind,
                native_wind_gust_speed=round(data['gust_max'], 1) if data['gust_max'] > 0 else None,
                cloud_coverage=avg_cloud,
                condition=condition,
                uv_index=round(data['uv_max'], 1) if data['uv_max'] > 0 else None,
            )

            forecasts.append(forecast)

        _LOGGER.info("Created %s daily forecast points", len(forecasts))
        return forecasts if forecasts else None

    def _select_daily_condition(self, conditions: list[dict]) -> str | None:
        """Select the most representative weather condition for the day."""
        if not conditions:
            return None

        # Define severity order
        severity_map = {
            1: 1, 2: 2, 3: 3, 4: 3, 5: 3,
            6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
        }

        # Score conditions
        scored_conditions = []
        for cond in conditions:
            code = cond['code']
            hour = cond['hour']
            precip = cond['precip']
            score = severity_map.get(code, 0) * 100
            if precip > 0:
                score += 50
            if 11 <= hour <= 15:
                score += 30
            elif 8 <= hour <= 18:
                score += 10
            scored_conditions.append({'code': code, 'score': score})

        scored_conditions.sort(key=lambda x: x['score'], reverse=True)
        best_condition_code = scored_conditions[0]['code']
        return WEATHER_SYMBOL_MAP.get(best_condition_code, "cloudy")

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()