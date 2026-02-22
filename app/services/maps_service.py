"""
Google Maps API integration for the Vehicle Emission Eco-Route System.

This module provides integration with Google Maps Directions API with
retry logic, exponential backoff, and in-memory caching.

Requirements: 3.1, 3.5, 3.6, 3.7, 14.4
"""

import time
import requests
from typing import Optional, List, Dict, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from app.core.exceptions import MapsAPIError, MapsAPITimeoutError, MapsAPIAuthError


class MapsService:
    """
    Google Maps API integration with retry and caching.
    
    Provides route retrieval with automatic retry on failures,
    exponential backoff, and in-memory caching for performance.
    
    Requirements: 3.1, 3.5, 3.6, 3.7, 14.4
    """
    
    def __init__(self, api_key: str, timeout: int = 10, max_retries: int = 3):
        """
        Initialize the Maps service.
        
        Args:
            api_key: Google Maps API key
            timeout: Request timeout in seconds (default: 10)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache: Dict[str, Tuple[dict, float]] = {}  # Cache: {key: (data, timestamp)}
        self.cache_ttl = 300  # 5 minutes in seconds
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.Timeout)
    )
    def get_routes(
        self,
        source: str,
        destination: str,
        alternatives: bool = True
    ) -> dict:
        """
        Get route alternatives from Google Maps API with retry logic.
        
        Implements exponential backoff retry (3 attempts, 2s/4s/8s delays)
        and in-memory caching with 5-minute TTL.
        
        Args:
            source: Source location (address or place name)
            destination: Destination location (address or place name)
            alternatives: Whether to request alternative routes (default: True)
            
        Returns:
            dict: Google Maps API response containing routes
            
        Raises:
            MapsAPITimeoutError: If request times out after all retries
            MapsAPIAuthError: If API key is invalid
            MapsAPIError: For other API errors
            
        Requirements: 3.1, 3.5, 3.6, 3.7, 14.4
        """
        # Check cache first
        cache_key = f"{source}|{destination}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            # Check if cache is still valid (within TTL)
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        # Prepare API request
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": source,
            "destination": destination,
            "alternatives": str(alternatives).lower(),
            "departure_time": "now",  # Request traffic data
            "key": self.api_key
        }
        
        try:
            # Make API request with timeout
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Check for API-level errors
            if data.get("status") == "REQUEST_DENIED":
                raise MapsAPIAuthError("Invalid Google Maps API key")
            
            if data.get("status") == "ZERO_RESULTS":
                # Return empty routes instead of raising error
                # This allows the caller to handle "no routes found" appropriately
                return {"routes": [], "status": "ZERO_RESULTS"}
            
            if data.get("status") != "OK" and data.get("status") != "ZERO_RESULTS":
                error_message = data.get("error_message", "Unknown error")
                raise MapsAPIError(f"Google Maps API error: {error_message}")
            
            # Cache the successful result
            self.cache[cache_key] = (data, time.time())
            
            return data
            
        except requests.exceptions.Timeout:
            # This will trigger retry via tenacity decorator
            raise MapsAPITimeoutError("Google Maps API request timed out")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401 or e.response.status_code == 403:
                raise MapsAPIAuthError("Invalid Google Maps API key")
            raise MapsAPIError(f"Google Maps API HTTP error: {e}")
            
        except requests.exceptions.RequestException as e:
            raise MapsAPIError(f"Google Maps API request failed: {e}")
    
    def parse_routes(self, api_response: dict) -> List[dict]:
        """
        Parse Google Maps API response into route data.
        
        Extracts relevant route information including distance, duration,
        traffic data, and route summary.
        
        Args:
            api_response: Raw Google Maps API response
            
        Returns:
            list[dict]: List of parsed route data dictionaries
            
        Requirements: 3.1
        """
        routes = api_response.get("routes", [])
        
        if not routes:
            return []
        
        parsed_routes = []
        for route in routes:
            # Get the first leg (assuming single-leg routes)
            leg = route["legs"][0]
            
            # Extract duration in traffic if available
            duration_in_traffic = None
            if "duration_in_traffic" in leg:
                duration_in_traffic = leg["duration_in_traffic"]["value"]
            
            parsed_route = {
                "distance_meters": leg["distance"]["value"],
                "distance_km": leg["distance"]["value"] / 1000,
                "duration_seconds": leg["duration"]["value"],
                "duration_minutes": leg["duration"]["value"] / 60,
                "duration_in_traffic": duration_in_traffic,
                "summary": route.get("summary", ""),
            }
            
            parsed_routes.append(parsed_route)
        
        return parsed_routes
