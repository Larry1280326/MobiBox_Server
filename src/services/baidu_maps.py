"""Baidu Maps API integration for reverse geocoding.

This module provides location enrichment via Baidu Maps API:
- Reverse geocoding: GPS coordinates -> location names
- POI information extraction
- Address formatting

Requires: Baidu Maps API key (set in BaiduMapsSettings)
"""

import asyncio
import logging
from typing import Optional

import aiohttp

from src.config import get_settings

logger = logging.getLogger(__name__)

# Baidu Maps API endpoint for reverse geocoding
BAIDU_REVERSE_GEOCODE_URL = "https://api.map.baidu.com/reverse_geocoding/v3/"

# Cache for location results (in-memory, simple TTL cache)
_location_cache: dict[str, dict] = {}
_cache_ttl_seconds = 3600  # 1 hour TTL


async def reverse_geocode(
    lat: float,
    lon: float,
    api_key: str | None = None,
) -> Optional[dict]:
    """Get location info from Baidu Maps reverse geocoding API.

    Converts GPS coordinates to human-readable location information.

    Args:
        lat: Latitude (WGS84 coordinate system)
        lon: Longitude (WGS84 coordinate system)
        api_key: Optional Baidu Maps API key (uses settings if not provided)

    Returns:
        Dict with location info:
        - address: Formatted address string
        - poi: List of POI names
        - business: Business area name
        - city: City name
        - district: District name
        Returns None if geocoding fails.
    """
    if api_key is None:
        settings = get_settings()
        api_key = settings.baidu_maps_api_key
        if not api_key:
            logger.debug("Baidu Maps API key not configured")
            return None

    # Check cache
    cache_key = f"{lat:.6f},{lon:.6f}"
    if cache_key in _location_cache:
        cached = _location_cache[cache_key]
        # Check TTL
        import time
        if time.time() - cached.get("_cached_at", 0) < _cache_ttl_seconds:
            logger.debug(f"Using cached location for {cache_key}")
            return {k: v for k, v in cached.items() if not k.startswith("_")}
        else:
            del _location_cache[cache_key]

    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "output": "json",
                "coordtype": "wgs84ll",  # WGS84 coordinate system
                "location": f"{lat},{lon}",
                "ak": api_key,
                "extensions_poi": "1",  # Include POI information
            }

            async with session.get(
                BAIDU_REVERSE_GEOCODE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Baidu Maps API returned status {resp.status}")
                    return None

                data = await resp.json()

                if data.get("status") != 0:
                    logger.warning(f"Baidu Maps API error: {data.get('message', 'Unknown error')}")
                    return None

                result = data.get("result", {})

                # Extract location components
                address_component = result.get("addressComponent", {})
                poi_regions = result.get("poiRegions", [])

                location_info = {
                    "address": result.get("formatted_address"),
                    "poi": [p.get("name") for p in poi_regions if p.get("name")],
                    "business": result.get("business"),
                    "city": address_component.get("city"),
                    "district": address_component.get("district"),
                    "province": address_component.get("province"),
                    "street": address_component.get("street"),
                    "street_number": address_component.get("street_number"),
                }

                # Cache the result
                import time
                location_info["_cached_at"] = time.time()
                _location_cache[cache_key] = location_info

                # Return without internal cache fields
                return {k: v for k, v in location_info.items() if not k.startswith("_")}

    except asyncio.TimeoutError:
        logger.warning(f"Baidu Maps API timeout for {lat},{lon}")
        return None
    except aiohttp.ClientError as e:
        logger.warning(f"Baidu Maps API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Baidu Maps API: {e}")
        return None


async def get_location_context(
    lat: float | None,
    lon: float | None,
    address: str | None = None,
    poi: str | None = None,
) -> Optional[str]:
    """Get enriched location context using Baidu Maps API.

    Combines provided address/POI with reverse geocoding to get
    accurate location context labels.

    Args:
        lat: Latitude (optional)
        lon: Longitude (optional)
        address: Provided address string (optional)
        poi: Provided POI string (optional)

    Returns:
        Location context string (e.g., "home", "work", "transit", etc.)
        or None if insufficient data.
    """
    # If we have coordinates, try Baidu Maps first
    if lat is not None and lon is not None:
        settings = get_settings()
        if settings.baidu_maps_enabled and settings.baidu_maps_api_key:
            location = await reverse_geocode(lat, lon)
            if location:
                # Combine Baidu results with provided data
                enriched_address = location.get("address") or address
                enriched_pois = location.get("poi", [])
                if poi and poi not in enriched_pois:
                    enriched_pois.append(poi)

                # Return combined info for LLM to analyze
                return {
                    "address": enriched_address,
                    "poi": ", ".join(enriched_pois) if enriched_pois else None,
                    "city": location.get("city"),
                    "district": location.get("district"),
                    "business": location.get("business"),
                }

    # Fallback: return provided info
    if address or poi:
        return {
            "address": address,
            "poi": poi,
        }

    return None


def clear_cache():
    """Clear the location cache."""
    global _location_cache
    _location_cache = {}