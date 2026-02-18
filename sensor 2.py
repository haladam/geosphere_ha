"""Sensor platform for GeoSphere Austria."""

from __future__ import annotations
from datetime import datetime
import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfTemperature,
    PERCENTAGE,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfPrecipitationDepth,
    UnitOfIrradiance,
    UnitOfLength,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import GeoSphereDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up GeoSphere sensors."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # Standard sensors for current conditions
    sensors = [
        GeoSphereSensor(coordinator, entry, "t2m", "Temperature"),
        GeoSphereSensor(coordinator, entry, "rh2m", "Humidity"),
        GeoSphereSensor(coordinator, entry, "rr_hourly", "Precipitation Rate"),
        GeoSphereSensor(coordinator, entry, "sp", "Pressure"),
        GeoSphereSensor(coordinator, entry, "wind_speed", "Wind Speed"),
        GeoSphereSensor(coordinator, entry, "wind_bearing", "Wind Bearing"),
        GeoSphereSensor(coordinator, entry, "wind_gust_speed", "Wind Gust Speed"),
        GeoSphereSensor(coordinator, entry, "tcc", "Cloud Coverage"),
        GeoSphereSensor(coordinator, entry, "grad_hourly", "Solar Radiation"),
        GeoSphereSensor(coordinator, entry, "snowlmt", "Snow Limit"),
        GeoSphereSensor(coordinator, entry, "rain_hourly", "Rain"),
        GeoSphereSensor(coordinator, entry, "snow_hourly", "Snow"),
        GeoSphereSensor(coordinator, entry, "cape", "CAPE"),
        GeoSphereSensor(coordinator, entry, "cin", "CIN"),
        GeoSphereSensor(coordinator, entry, "sy", "Weather Symbol"),
        GeoSphereSensor(coordinator, entry, "sundur_hourly", "Sunshine Duration"),
    ]

    # Add the 15-minute forecast sensor
    sensors.append(GeoSphere15MinForecastSensor(coordinator, entry))

    async_add_entities(sensors)


class GeoSphereSensor(CoordinatorEntity, SensorEntity):
    """GeoSphere sensor for current conditions."""

    def __init__(
        self,
        coordinator: GeoSphereDataUpdateCoordinator,
        entry: ConfigEntry,
        parameter: str,
        name: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._parameter = parameter
        self._attr_name = f"{entry.data['name']} {name}"
        self._attr_unique_id = f"{entry.entry_id}_{parameter}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.data["name"],
            "manufacturer": "GeoSphere Austria",
            "model": "Multi-Model (INCA Nowcast 1km + AROME 2.5km)",
            "entry_type": "service",
        }

        # Set device class and unit based on parameter
        self._set_sensor_properties()

    def _set_sensor_properties(self) -> None:
        """Set sensor properties based on parameter type."""
        if self._parameter in ["t2m", "mnt2m", "mxt2m"]:
            self._attr_device_class = SensorDeviceClass.TEMPERATURE
            self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif self._parameter in ["rh2m"]:
            self._attr_device_class = SensorDeviceClass.HUMIDITY
            self._attr_native_unit_of_measurement = PERCENTAGE
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif self._parameter in ["sp"]:
            self._attr_device_class = SensorDeviceClass.ATMOSPHERIC_PRESSURE
            self._attr_native_unit_of_measurement = UnitOfPressure.HPA
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif self._parameter in ["wind_speed", "wind_gust_speed"]:
            self._attr_device_class = SensorDeviceClass.WIND_SPEED
            self._attr_native_unit_of_measurement = UnitOfSpeed.METERS_PER_SECOND
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif self._parameter in ["wind_bearing", "dd"]:
            self._attr_native_unit_of_measurement = "°"
            self._attr_state_class = SensorStateClass.MEASUREMENT
            self._attr_icon = "mdi:compass"
        elif self._parameter in ["rr_hourly", "rain_hourly", "snow_hourly"]:
            self._attr_device_class = SensorDeviceClass.PRECIPITATION
            self._attr_native_unit_of_measurement = UnitOfPrecipitationDepth.MILLIMETERS
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif self._parameter in ["grad_hourly"]:
            self._attr_device_class = SensorDeviceClass.IRRADIANCE
            self._attr_native_unit_of_measurement = UnitOfIrradiance.WATTS_PER_SQUARE_METER
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif self._parameter in ["tcc"]:
            self._attr_native_unit_of_measurement = PERCENTAGE
            self._attr_state_class = SensorStateClass.MEASUREMENT
            self._attr_icon = "mdi:cloud-percent"
        elif self._parameter in ["snowlmt"]:
            self._attr_device_class = SensorDeviceClass.DISTANCE
            self._attr_native_unit_of_measurement = UnitOfLength.METERS
            self._attr_state_class = SensorStateClass.MEASUREMENT
            self._attr_icon = "mdi:snowflake-melt"
        elif self._parameter in ["cape", "cin"]:
            self._attr_native_unit_of_measurement = "J/kg"
            self._attr_icon = "mdi:weather-lightning" if self._parameter == "cape" else "mdi:arrow-down-bold"
        elif self._parameter in ["sy"]:
            self._attr_icon = "mdi:weather-partly-cloudy"
        elif self._parameter in ["sundur_hourly"]:
            self._attr_native_unit_of_measurement = "min"
            self._attr_icon = "mdi:weather-sunny"

    @property
    def native_value(self) -> float | None:
        """Return the state of the sensor."""
        if not self.coordinator.data:
            return None

        current = self.coordinator.data.get("current", {})
        value = current.get(self._parameter)

        # Convert pressure from Pa to hPa
        if self._parameter == "sp" and value is not None:
            value = round(value / 100, 1)

        # Convert cloud coverage from fraction to percentage
        if self._parameter == "tcc" and value is not None:
            value = round(value * 100)

        # Convert solar radiation from J to W/m²
        if self._parameter == "grad_hourly" and value is not None:
            value = round(value / 3600)

        # Convert sunshine duration from seconds to minutes
        if self._parameter == "sundur_hourly" and value is not None:
            value = round(value / 60, 1)

        # Round other values
        if value is not None and self._parameter not in ["sy", "wind_bearing", "dd"]:
            if isinstance(value, float):
                value = round(value, 2)

        return value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        if not self.coordinator.data:
            return {}

        return {
            "reference_time": self.coordinator.data.get("reference_time"),
            "last_update": self.coordinator.data.get("current_timestamp"),
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()


class GeoSphere15MinForecastSensor(CoordinatorEntity, SensorEntity):
    """Sensor that exposes 15-minute forecast data for visualization."""

    def __init__(
        self,
        coordinator: GeoSphereDataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_name = f"{entry.data['name']} 15-Min Forecast"
        self._attr_unique_id = f"{entry.entry_id}_15min_forecast"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.data["name"],
            "manufacturer": "GeoSphere Austria",
            "model": "Multi-Model (INCA Nowcast 1km + AROME 2.5km)",
            "entry_type": "service",
        }
        self._attr_icon = "mdi:chart-line"

    @property
    def native_value(self) -> str:
        """Return the number of forecast points available."""
        if not self.coordinator.data or not self.coordinator.data.get("forecast"):
            return "0"

        forecast_count = len(self.coordinator.data["forecast"])
        return str(forecast_count)

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return "points"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return forecast data as attributes for visualization."""
        if not self.coordinator.data or not self.coordinator.data.get("forecast"):
            return {}

        forecast_data = self.coordinator.data["forecast"]

        timestamps = []
        temperatures = []
        precipitation = []
        humidity = []
        wind_speed = []
        wind_gust = []
        wind_bearing = []
        pressure = []
        cloud_cover = []
        weather_symbols = []
        sources = []

        # Extract first 48 points (12 hours at 15-min intervals)
        for point in forecast_data[:48]:
            dt_str = point.get("datetime")
            if not dt_str:
                continue

            try:
                dt = datetime.fromisoformat(dt_str.replace("+00:00", "+00:00"))
                timestamps.append(dt.isoformat())

                temp = point.get("t2m")
                temperatures.append(round(temp, 1) if temp is not None else None)

                precip = point.get("rr_hourly", 0)
                precipitation.append(round(precip, 2) if precip is not None else 0)

                rh = point.get("rh2m")
                humidity.append(round(rh, 1) if rh is not None else None)

                ws = point.get("wind_speed")
                wind_speed.append(round(ws * 3.6, 1) if ws is not None else None)

                wg = point.get("wind_gust_speed")
                wind_gust.append(round(wg * 3.6, 1) if wg is not None else None)

                wb = point.get("wind_bearing")
                wind_bearing.append(round(wb) if wb is not None else None)

                p = point.get("sp")
                pressure.append(round(p / 100, 1) if p is not None else None)

                cc = point.get("tcc")
                cloud_cover.append(round(cc * 100) if cc is not None else None)

                sy = point.get("sy")
                weather_symbols.append(int(sy) if sy is not None else None)

                source = point.get("source", "unknown")
                sources.append(source)

            except (ValueError, TypeError) as err:
                _LOGGER.debug("Error processing forecast point: %s", err)
                continue

        return {
            "reference_time": self.coordinator.data.get("reference_time"),
            "inca_last_update": self.coordinator.data.get("last_api_update_inca"),
            "arome_last_update": self.coordinator.data.get("last_api_update_arome"),
            "forecast_points": len(timestamps),
            "timestamps": timestamps,
            "temperature": temperatures,
            "precipitation": precipitation,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_gust": wind_gust,
            "wind_bearing": wind_bearing,
            "pressure": pressure,
            "cloud_cover": cloud_cover,
            "weather_symbols": weather_symbols,
            "data_sources": sources,
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()